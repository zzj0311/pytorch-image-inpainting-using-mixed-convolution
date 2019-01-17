import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as opt
import torch.utils.data as data
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.utils import make_grid
from torchvision.utils import save_image

#def weights_init(m, nonlinearity='relu'):
#    if isinstance(m, nn.Conv2d):
#        nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)

def get_state_dict_on_cpu(obj):
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))
    return state_dict

def save_ckpt(ckpt_name, models, optimizers, n_epoch):
    ckpt_dict = {'n_epoch': n_epoch}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)

    for prefix, optimizer in optimizers:
        ckpt_dict[prefix] = optimizer.state_dict()
    torch.save(ckpt_dict, ckpt_name)

def load_ckpt(ckpt_name, models, optimizers=None):
    ckpt_dict = torch.load(ckpt_name)
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])
    return ckpt_dict['n_epoch']

def unnormalize(x, std, mean):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(std) + torch.Tensor(mean)
    x = x.transpose(1, 3)
    return x

def weights_init(nonlinearity='relu'):
    '''
        Inital weights using Kaiming He's method in "Delving deep into rectifiers:
        Surpassing human-level performance on ImageNet classification"

        Input:
            non-linearity activation type, should be a string in {'relu', 'leaky_relu'}
        
        Return:
            The initial func.
        
        Modified from https://github.com/mingyuliutw/UNIT/blob/master/utils.py
    '''
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0) and hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    return init_fun
    
def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
            torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss

class InpaintingLoss(nn.Module):
    def __init__(self, extractor):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, input, mask, output, gt):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output

        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)

        if output.shape[1] == 3:
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([output_comp]*3, 1))
            feat_output = self.extractor(torch.cat([output]*3, 1))
            feat_gt = self.extractor(torch.cat([gt]*3, 1))
        else:
            raise ValueError('only gray and RGB img is supported')

        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))

        loss_dict['tv'] = total_variation_loss(output_comp)

        return loss_dict

def evaluate(model, dataset, device, filename, std, mean, count):
    image, mask, gt = zip(*[dataset[i] for i in range(count)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _, learned_mask = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    learned_mask = learned_mask.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    image_o = unnormalize(image, std, mean)
    output_o = unnormalize(output, std, mean)
    output_comp_o = unnormalize(output_comp, std, mean)
    gt_o = unnormalize(gt, std, mean)

    grid = make_grid(
        torch.cat(
                (
                    image_o, \
                    output_o, \
                    output_comp_o, \
                    gt_o, \
                    torch.cat((mask, mask, mask), dim=1), \
                    torch.cat((learned_mask, learned_mask, learned_mask), dim=1) \
                ), dim=0) \
                    )
    save_image(grid, filename)

def train(loader_train, loader_val, dSet_val, model, optimizer, criterion, start_epoch, epoch, \
            interval, device, loss_config, dSet_std, dSet_mean, save_dir):

    loss_list = []
    thetime = 0
    for e in range(start_epoch, epoch):
        logging.info("-----------epoch %s start here----------", e)
        print("-----------epoch {} start here----------".format(e))
        for t, (image, mask, gt) in enumerate(loader_train):
            start = time.time()
            model.train()
            image = image.to(device)
            mask = mask.to(device)
            gt = gt.to(device)

            output, _, _ = model(image, mask)
            loss_dict = criterion(image, mask, output, gt)

            loss = sum((loss_dict[k] * v for k, v in loss_config.items()))
            loss_list.append(loss_dict)
            logging.info("loss: %s", loss_dict)
            print(loss_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end = time.time()
            thetime += end - start

            if t+1 % interval == 0:
                logging.info("iter %s \n, loss: %s, loss_dict: %s", t, loss, loss_dict)
                model.eval()
                evaluate(model, dSet_val, device,
                        '{:s}/images/test_{:d}_{:d}.jpg'.format(save_dir, e, t), dSet_std, dSet_mean, 8)
        
        if e % 10 == 0:
            save_ckpt('{:s}/ckpt/{:d}.pth'.format(save_dir, e), [('model', model)], [('optimizer', optimizer)], e)

    save_ckpt('{:s}/ckpt/{:s}.pth'.format(save_dir, 'final'), [('model', model)], [('optimizer', optimizer)], e)
    return loss_list, thetime / t

def modelParam(model):
    return sum((x.numel() for x in model.parameters()))

def modelTest(modelList, loader_test, device, savedir, std, mean):
    timeDict = {"time{}".format(x):[] for x in range(len(modelList))}
    lossDict = {"loss{}".format(x):{'l1':[], 'l2':[], 'tv':[]} for x in range(len(modelList))}
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()
    for t, (image, mask, gt) in enumerate(loader_test):
        theList = [unnormalize(image, std, mean)]

        for index, model in enumerate(modelList):
            start = time.time()
            with torch.no_grad():
                outputemp, _, _ = model(image.to(device), mask.to(device))
            outputemp = outputemp.to(torch.device('cpu'))
            lossDict["loss{}".format(index)]['l1'].append(l1(outputemp, gt).data.numpy())
            lossDict["loss{}".format(index)]['l2'].append(l2(outputemp, gt).data.numpy())
            lossDict["loss{}".format(index)]['tv'].append(total_variation_loss(outputemp).data.numpy())
            theList.append(unnormalize(outputemp, std, mean))
            end = time.time()
            timeDict["time{}".format(index)].append(end - start)

        theList.append(unnormalize(gt, std, mean))

        grid = make_grid(torch.cat(theList))
        save_image(grid, "{}/{}.jpg".format(savedir, t))
    return {k:(sum(v) / len(v)) for k, v in timeDict.items()}, {k:{kk:np.mean(vv) for kk, vv in v.items()} for k, v in lossDict.items()}