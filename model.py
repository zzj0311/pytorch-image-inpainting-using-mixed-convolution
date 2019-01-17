from util import *
from torchvision import models
import random

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

class PartialConv2d(nn.Conv2d):
    '''
        Partial conv layer for 2d matrix. Derived from nn.Conv2d.

        Modified from https://github.com/NVIDIA/partialconv/blob/master/models/partialconv2d.py
    '''
    def __init__(self, *args, **kwargs):

        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'bias' in kwargs:
            self.bias = kwargs['bias']
            kwargs.pop('bias')
        else:
            self.multi_channel = False

        super(PartialConv2d, self).__init__(*args, **kwargs)
        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None)
        self.update_mask = None
        self.mask_ratio = None
        
    def forward(self, input, mask=None):
        if mask is not None or self.last_size != (input.data.shape[2], input.data.shape[3]):
            self.last_size = (input.data.shape[2], input.data.shape[3])

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)
                        
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1) # work as sum(slide_win) > 0
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
            self.update_mask.to(input)
            self.mask_ratio.to(input)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask))

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        return output, self.update_mask

class gatedConv2dWithActivation(torch.nn.Module):
    '''
    Gated Convlution layer with activation (default activation:LeakyReLU)

    Input:
        The feature from last layer
    return:
        \phi(f(I))*\sigmoid(g(I))
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=True, update_mask = False, activation='relu'):
        super(gatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.update_mask = update_mask
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        if self.activation != None:
            self.conv2d.apply(weights_init(nonlinearity=self.activation))
            self.mask_conv2d.apply(weights_init(nonlinearity=self.activation))
        else:
            self.conv2d.apply(weights_init)
            self.mask_conv2d.apply(weights_init)
        self.batch_norm2d.apply(weights_init)

    def gated(self, mask):
        return self.sigmoid(mask)

    def maskOut(self, mask):
        return (self.gated(mask) < 0.5).to(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation == 'relu':
            activ = nn.ReLU()
            x = activ(x) * self.gated(mask)
        elif self.activation == 'leaky_relu':
            activ = nn.LeakyReLU(negative_slope=0.2)
            x = activ(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)

        if self.batch_norm:
            x = self.batch_norm2d(x)
        else:
            x = x
        if self.update_mask:
            return x, self.maskOut(mask)
        else:
            return x, mask

class gatedDeconv2dWithActivation(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True,activation='relu', update_mask=False):
        super(gatedDeconv2dWithActivation, self).__init__()
        self.conv2d = gatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, update_mask, activation)
        self.scale_factor = scale_factor

    def forward(self, input1, input2):
        input1 = F.interpolate(input1, scale_factor=self.scale_factor)
        return self.conv2d(torch.cat((input1, input2), dim=1))

class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='same-3', activ='relu',
                 conv_bias=False, multi_channel=True):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv2d(in_ch, out_ch, 5, 2, 2, bias=conv_bias, multi_channel=multi_channel)
        elif sample == 'down-7':
            self.conv = PartialConv2d(in_ch, out_ch, 7, 2, 3, bias=conv_bias, multi_channel=multi_channel)
        elif sample == 'down-3':
            self.conv = PartialConv2d(in_ch, out_ch, 3, 2, 1, bias=conv_bias, multi_channel=multi_channel)
        else:
            self.conv = PartialConv2d(in_ch, out_ch, 3, 1, 1, bias=conv_bias, multi_channel=multi_channel)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
            self.bn.apply(weights_init())

        if activ == 'relu':
            self.conv.apply(weights_init(nonlinearity='relu'))
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.conv.apply(weights_init(nonlinearity='leaky_relu'))
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask

class PConvUNet(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, mask_channels=1, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        self.enc_5 = PCBActiv(512, 512, sample='down-3')
        self.enc_6 = PCBActiv(512, 512, sample='down-3')
        self.enc_7 = PCBActiv(512, 512, sample='down-3')

        self.dec_7 = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_6 = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_5 = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, input_channels, bn=False, activ=None, conv_bias=True)

    def forward(self, input, input_mask):        
        h_input_1, h_mask_1 = self.enc_1(input, torch.cat((input_mask, input_mask, input_mask), dim=1))
        h_input_2, h_mask_2 = self.enc_2(h_input_1, h_mask_1)
        h_input_3, h_mask_3 = self.enc_3(h_input_2, h_mask_2)
        h_input_4, h_mask_4 = self.enc_4(h_input_3, h_mask_3)
        h_input_5, h_mask_5 = self.enc_5(h_input_4, h_mask_4)
        h_input_6, h_mask_6 = self.enc_6(h_input_5, h_mask_5)
        h_input_7, h_mask_7 = self.enc_7(h_input_6, h_mask_6)

        h_input = F.interpolate(h_input_7, scale_factor=2, mode=self.upsampling_mode)
        h_mask = F.interpolate(h_mask_7, scale_factor=2, mode=self.upsampling_mode)
        h_input, h_mask = self.dec_7(torch.cat((h_input, h_input_6), dim=1), torch.cat((h_mask, h_mask_6), dim=1))

        h_input = F.interpolate(h_input, scale_factor=2, mode=self.upsampling_mode)
        h_mask = F.interpolate(h_mask, scale_factor=2, mode=self.upsampling_mode)
        h_input, h_mask = self.dec_6(torch.cat((h_input, h_input_5), dim=1), torch.cat((h_mask, h_mask_5), dim=1))

        h_input = F.interpolate(h_input, scale_factor=2, mode=self.upsampling_mode)
        h_mask = F.interpolate(h_mask, scale_factor=2, mode=self.upsampling_mode)
        h_input, h_mask = self.dec_5(torch.cat((h_input, h_input_4), dim=1), torch.cat((h_mask, h_mask_4), dim=1))

        h_input = F.interpolate(h_input, scale_factor=2, mode=self.upsampling_mode)
        h_mask = F.interpolate(h_mask, scale_factor=2, mode=self.upsampling_mode)
        h_input, h_mask = self.dec_4(torch.cat((h_input, h_input_3), dim=1), torch.cat((h_mask, h_mask_3), dim=1))

        h_input = F.interpolate(h_input, scale_factor=2, mode=self.upsampling_mode)
        h_mask = F.interpolate(h_mask, scale_factor=2, mode=self.upsampling_mode)
        h_input, h_mask = self.dec_3(torch.cat((h_input, h_input_2), dim=1), torch.cat((h_mask, h_mask_2), dim=1))

        h_input = F.interpolate(h_input, scale_factor=2, mode=self.upsampling_mode)
        h_mask = F.interpolate(h_mask, scale_factor=2, mode=self.upsampling_mode)
        h_input, h_mask = self.dec_2(torch.cat((h_input, h_input_1), dim=1), torch.cat((h_mask, h_mask_1), dim=1))

        h_input = F.interpolate(h_input, scale_factor=2, mode=self.upsampling_mode)
        h_mask = F.interpolate(h_mask, scale_factor=2, mode=self.upsampling_mode)

        h, h_mask = self.dec_1(torch.cat((h_input, input), dim=1), \
                                torch.cat((h_mask, torch.cat((input_mask, input_mask, input_mask), dim=1)), dim=1))

        return h, h_mask[:, 0, :, :].unsqueeze(dim=1), h_mask[:, 0, :, :].unsqueeze(dim=1)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    for param in module.parameters():
                        param.requires_grad = False

class PConvUNet3(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        #self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_1 = gatedConv2dWithActivation(input_channels, 64, 7, 2, 3, bias=True, batch_norm=False, update_mask=True)
        self.enc_2 = gatedConv2dWithActivation(64, 128, 5, 2, 2, bias=True, batch_norm=True, update_mask=True)
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        self.enc_5 = PCBActiv(512, 512, sample='down-3')
        self.enc_6 = PCBActiv(512, 512, sample='down-3')
        self.enc_7 = PCBActiv(512, 512, sample='down-3')

        self.dec_7 = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_6 = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_5 = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = gatedConv2dWithActivation(64 + input_channels, input_channels, 3, 1, 1, batch_norm=False, activation=None, update_mask=True, bias=True)

    def forward(self, input, input_mask):        
        h_input_1, h_mask_1 = self.enc_1(input)
        h_input_2, h_mask_2 = self.enc_2(h_input_1)

        layerNum = random.randint(0, 127)
        l_mask = F.interpolate(h_mask_2, scale_factor=4, mode=self.upsampling_mode)
        l_mask = l_mask[:, layerNum, :, :].unsqueeze(1)
        h_input_3, h_mask_3 = self.enc_3(h_input_2, h_mask_2)
        h_input_4, h_mask_4 = self.enc_4(h_input_3, h_mask_3)
        h_input_5, h_mask_5 = self.enc_5(h_input_4, h_mask_4)
        h_input_6, h_mask_6 = self.enc_6(h_input_5, h_mask_5)
        h_input_7, h_mask_7 = self.enc_7(h_input_6, h_mask_6)

        h_input = F.interpolate(h_input_7, scale_factor=2, mode=self.upsampling_mode)
        h_mask = F.interpolate(h_mask_7, scale_factor=2, mode=self.upsampling_mode)
        h_input, h_mask = self.dec_7(torch.cat((h_input, h_input_6), dim=1), torch.cat((h_mask, h_mask_6), dim=1))

        h_input = F.interpolate(h_input, scale_factor=2, mode=self.upsampling_mode)
        h_mask = F.interpolate(h_mask, scale_factor=2, mode=self.upsampling_mode)
        h_input, h_mask = self.dec_6(torch.cat((h_input, h_input_5), dim=1), torch.cat((h_mask, h_mask_5), dim=1))

        h_input = F.interpolate(h_input, scale_factor=2, mode=self.upsampling_mode)
        h_mask = F.interpolate(h_mask, scale_factor=2, mode=self.upsampling_mode)
        h_input, h_mask = self.dec_5(torch.cat((h_input, h_input_4), dim=1), torch.cat((h_mask, h_mask_4), dim=1))

        h_input = F.interpolate(h_input, scale_factor=2, mode=self.upsampling_mode)
        h_mask = F.interpolate(h_mask, scale_factor=2, mode=self.upsampling_mode)
        h_input, h_mask = self.dec_4(torch.cat((h_input, h_input_3), dim=1), torch.cat((h_mask, h_mask_3), dim=1))

        h_input = F.interpolate(h_input, scale_factor=2, mode=self.upsampling_mode)
        h_mask = F.interpolate(h_mask, scale_factor=2, mode=self.upsampling_mode)
        h_input, h_mask = self.dec_3(torch.cat((h_input, h_input_2), dim=1), torch.cat((h_mask, h_mask_2), dim=1))

        h_input = F.interpolate(h_input, scale_factor=2, mode=self.upsampling_mode)
        h_mask = F.interpolate(h_mask, scale_factor=2, mode=self.upsampling_mode)
        h_input, h_mask = self.dec_2(torch.cat((h_input, h_input_1), dim=1), torch.cat((h_mask, h_mask_1), dim=1))

        h_input = F.interpolate(h_input, scale_factor=2, mode=self.upsampling_mode)
        h_mask = F.interpolate(h_mask, scale_factor=2, mode=self.upsampling_mode)

        h, h_mask = self.dec_1(torch.cat((h_input, input), dim=1))

        return h, h_mask, l_mask
    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    for param in module.parameters():
                        param.requires_grad = False

class GConvUNet(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, mask_channels=1, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = gatedConv2dWithActivation(input_channels, 64, 7, 2, 3, bias=True, batch_norm=False)
        self.enc_2 = gatedConv2dWithActivation(64, 128, 5, 2, 2, bias=True, batch_norm=True)
        self.enc_3 = gatedConv2dWithActivation(128, 256, 5, 2, 2, bias=True, batch_norm=True)
        self.enc_4 = gatedConv2dWithActivation(256, 512, 3, 2, 1, bias=True, batch_norm=True)
        self.enc_5 = gatedConv2dWithActivation(512, 512, 3, 2, 1, bias=True, batch_norm=True)
        self.enc_6 = gatedConv2dWithActivation(512, 512, 3, 2, 1, bias=True, batch_norm=True)
        self.enc_7 = gatedConv2dWithActivation(512, 512, 3, 2, 1, bias=True, batch_norm=True)

        self.dec_7 = gatedDeconv2dWithActivation(2, 512 + 512, 512, 3, 1, 1, bias=True, batch_norm=True, activation='leaky_relu')
        self.dec_6 = gatedDeconv2dWithActivation(2, 512 + 512, 512, 3, 1, 1, bias=True, batch_norm=True, activation='leaky_relu')
        self.dec_5 = gatedDeconv2dWithActivation(2, 512 + 512, 512, 3, 1, 1, bias=True, batch_norm=True, activation='leaky_relu')
        self.dec_4 = gatedDeconv2dWithActivation(2, 512 + 256, 256, 3, 1, 1, bias=True, batch_norm=True, activation='leaky_relu')
        self.dec_3 = gatedDeconv2dWithActivation(2, 256 + 128, 128, 3, 1, 1, bias=True, batch_norm=True, activation='leaky_relu')
        self.dec_2 = gatedDeconv2dWithActivation(2, 128 + 64, 64, 3, 1, 1, bias=True, batch_norm=True, activation='leaky_relu')
        self.dec_1 = gatedDeconv2dWithActivation(2, 64 + input_channels, input_channels, 3, 1, 1, batch_norm=False, activation=None, update_mask=True, bias=True)

    def forward(self, input, input_mask):        
        h_input_1, _ = self.enc_1(input)
        h_input_2, _ = self.enc_2(h_input_1)
        h_input_3, _ = self.enc_3(h_input_2)
        h_input_4, _ = self.enc_4(h_input_3)
        h_input_5, _ = self.enc_5(h_input_4)
        h_input_6, _ = self.enc_6(h_input_5)
        h_input_7, _ = self.enc_7(h_input_6)

        h_input, _ = self.dec_7(h_input_7, h_input_6)

        h_input, _ = self.dec_6(h_input, h_input_5)

        h_input, _ = self.dec_5(h_input, h_input_4)

        h_input, _ = self.dec_4(h_input, h_input_3)

        h_input, _ = self.dec_3(h_input, h_input_2)

        h_input, _ = self.dec_2(h_input, h_input_1)

        h, h_mask = self.dec_1(h_input, input)

        return h, h_mask[:, 0, :, :].unsqueeze(dim=1), h_mask[:, 0, :, :].unsqueeze(dim=1)
    
    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    for param in module.parameters():
                        param.requires_grad = False

model = {   
            "pconv3":PConvUNet3(),
            "pconv" :PConvUNet(),
            "gconv" :GConvUNet()
        }

def getModel(tag):
    return model[tag]
