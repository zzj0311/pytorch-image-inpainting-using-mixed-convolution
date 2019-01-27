import argparse
import os
import time

from util import *
from celebA import *
from model import getModel, VGG16FeatureExtractor

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--dset_path', '-s', type=str, default='./img_align_celeba_flist.pkl')
parser.add_argument('--mask_path', '-m', type=str, default='./val_mask.pkl')
parser.add_argument('--save_dir', '-d', type=str, default='./modelSave')
parser.add_argument('--log_dir', type=str, default='./log.txt')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr_finetune', type=float, default=5e-5)
parser.add_argument('--epoch', '-e', type=int, default=100)
parser.add_argument('--batch_size', '-b', type=int, default=1)
parser.add_argument('--n_threads', '-n', type=int, default=16)
parser.add_argument('--interval', '-I', type=int, default=50)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--resume', '-r', type=str, default=None)
parser.add_argument('--finetune', '-f', action='store_true')
parser.add_argument('--device', '-i', type=str, default='0')
parser.add_argument('--model_tag', '-t', type=str, default='pconv')
args = parser.parse_args()

logging.basicConfig(level=0,\
                    format='%(asctime)s %(filename)s[line:%(lineno)d] : %(message)s',\
                    datefmt='%a, %d %b %Y %H:%M:%S',\
                    filename=args.log_dir,\
                    filemode="a+")

if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/images'.format(args.save_dir))
    os.makedirs('{:s}/ckpt'.format(args.save_dir))

paths = pickle.load(open(args.dset_path, 'rb'))
dSet_mean = paths['mean']
dSet_std = paths['std']
device = torch.device('cuda:{}'.format(args.device))

img_tf = [T.Resize( size=( args.image_size, args.image_size ) ), T.ToTensor(), T.Normalize(mean=dSet_mean, std=dSet_std)]
mask_tf = [T.ToTensor()]

dataset_train = celebA(args.dset_path, img_tf, mask_tf, size=( args.image_size, args.image_size ))
dataset_val = celebA(args.dset_path, img_tf, mask_tf, train='val', maskDumpFile=args.mask_path, size=( args.image_size, args.image_size ))
dataset_test = celebA(args.dset_path, img_tf, mask_tf, train='test', size=( args.image_size, args.image_size ))

dSet_len = len(dataset_train)
print((dSet_std, dSet_mean))

NUM_TRAIN = dSet_len - 12

loader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, 
                          sampler=data.sampler.SubsetRandomSampler(range(NUM_TRAIN)))
loader_val = data.DataLoader(dataset_val, batch_size=args.batch_size, 
                        sampler=data.sampler.SubsetRandomSampler(range(NUM_TRAIN, dSet_len)))
loader_test = data.DataLoader(dataset_test, batch_size=args.batch_size)

logging.info("using device:{}".format(device))

model = getModel(args.model_tag)
model = model.to(device)

if args.finetune:
    lr = args.lr_finetune
    model.freeze_enc_bn = True
else:
    lr = args.lr
loss_config = {'valid': 1.0, \
                'hole': 6.0, \
                'tv': 0.1, \
                'prc': 0.05, \
                'style': 120.0 \
            }

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)

if args.resume:
    start_epoch = load_ckpt(args.resume, [('model', model)], [('optimizer', optimizer)])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print('Starting from iter ', start_epoch)
else:
    start_epoch = 0

loss, time = train(loader_train, loader_val, dataset_val, model, optimizer, criterion, start_epoch=start_epoch, epoch=args.epoch, \
        interval=args.interval, device=device, loss_config=loss_config, dSet_std=dSet_std, 
        dSet_mean=dSet_mean, save_dir=args.save_dir)

print(time)
#evaluate(model, dataset_test, device, '{:s}/images/test_{:s}.jpg'.format(args.save_dir, 'final'), dSet_std, dSet_mean, 8)

