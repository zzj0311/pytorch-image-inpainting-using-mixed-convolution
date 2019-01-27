import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image
import pickle
import numpy as np
import cv2
import random

def random_ff_mask(configDict=None, size = (256, 256)):
    """
    Generate a random free form mask according to "Free-Form Image Inpainting with Gated Convolution".
    With default setting, around 1/5 is covered by mask, on average.

    Args:
        a dict of mask config with following attrs:
        -- shape: a sequence of mask shape
        -- vertex: num of vertex range
        -- length: trace length range
        -- bWid: brush width range
        -- angle: angle change in rad range
        -- cnt: num of traces range
        size -- tuple of expected mask size

    Returns:
        np.ndarray random free form mask
    
    Modified from https://github.com/avalonstrel/GatedConvolution_pytorch
    """

    if configDict == None:
        configDict = {'shape':(256, 256), \
                        'vertex':20, \
                        'length':50, \
                        'bWid':20, \
                        'angle':4.0, \
                        'cnt':5 }

    h,w = configDict['shape']
    mask = np.zeros((h,w))
    num_c = random.randint(1, configDict['cnt'])

    for _ in range(num_c):
        start_x = random.randint(1, w)
        start_y = random.randint(1, h)
        num_v = random.randint(10, configDict['vertex'])
        for i in range(num_v):
            angle = 0.01+np.random.randint(configDict['angle'])
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = random.randint(10, configDict['length'])
            brush_w = random.randint(10, configDict['bWid'])
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)

            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
            start_x, start_y = end_x, end_y

    mask = cv2.resize(mask, size, cv2.INTER_NEAREST)
    return mask.reshape(mask.shape+(1,)).astype('uint8') * 255

class celebA(data.Dataset):
    def __init__(self, pathFile, img_transform, mask_transform, maskDumpFile=None, train='train', mask_config=None, size=(256, 256)):
        super().__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.train = train
        self.mask_config = mask_config
        self.size = size

        filePathDict = pickle.load(open(pathFile, "rb"))
        if self.train == 'train':
            self.paths = filePathDict['train']
            self.maskList = None
        elif self.train == 'test':
            self.paths = filePathDict['test']
            self.maskList = None
        elif self.train == 'val' and maskDumpFile != None:
            self.paths = filePathDict['train']
            self.maskList = pickle.load(open(maskDumpFile, "rb"))
        else:
            raise("DsetTagError")

    def __getitem__(self, index):
        gt = Image.open(self.paths[index])
        gt = T.Compose(self.img_transform)(gt.convert('RGB'))
        if self.train == 'train' or self.train == 'test' or self.maskList == None:
            mask = random_ff_mask(self.mask_config, self.size)
        else:
            mask = cv2.resize(self.maskList[index % 16], self.size, cv2.INTER_NEAREST)
        mask = T.Compose(self.mask_transform)(255-mask)
        return gt * mask, mask, gt
        
    def __len__(self):
        return len(self.paths)