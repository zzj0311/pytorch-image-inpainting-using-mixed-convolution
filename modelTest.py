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
parser.add_argument('--Gpth', '-G', type=str, default='./modelSaveG/ckpt/final.pth')
parser.add_argument('--Opth', '-O', type=str, default='./modelSaveOrigin/ckpt/final.pth')
parser.add_argument('--Ppth', '-P', type=str, default='./modelSave3/ckpt/final.pth')
parser.add_argument('--log_dir', type=str, default='./log.txt')
parser.add_argument('--save_dir', type=str, default='./modelResult')
parser.add_argument('--batch_size', '-b', type=int, default=1)
parser.add_argument('--n_threads', '-n', type=int, default=2)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--device', '-i', type=str, default='0')
args = parser.parse_args()

logging.basicConfig(level=0,\
                    format='%(asctime)s %(filename)s[line:%(lineno)d] : %(message)s',\
                    datefmt='%a, %d %b %Y %H:%M:%S',\
                    filename=args.log_dir,\
                    filemode="a+")

paths = pickle.load(open(args.dset_path, 'rb'))
dSet_mean = paths['mean']
dSet_std = paths['std']
device = torch.device('cuda:{}'.format(args.device))

img_tf = [T.Resize( size=( args.image_size, args.image_size ) ), T.ToTensor(), T.Normalize(mean=dSet_mean, std=dSet_std)]
mask_tf = [T.ToTensor()]

dataset_test = celebA(args.dset_path, img_tf, mask_tf, train='test')
loader_test = data.DataLoader(dataset_test, batch_size=args.batch_size)

modelG = getModel("gconv")
modelG = modelG.to(device)

modelO = getModel("pconv")
modelO = modelO.to(device)

modelP = getModel("pconv3")
modelP = modelP.to(device)

_ = load_ckpt(args.Gpth, [('model', modelG)])
print("modelG param count: {}".format(modelParam(modelG) / 1e6))
_ = load_ckpt(args.Opth, [('model', modelO)])
print("modelO param count: {}".format(modelParam(modelO) / 1e6))
_ = load_ckpt(args.Ppth, [('model', modelP)])
print("modelP param count: {}".format(modelParam(modelP) / 1e6))

modelG.eval()
modelO.eval()
modelP.eval()

thetime, theloss = modelTest([modelO, modelG, modelP], loader_test, device, args.save_dir, dSet_std, dSet_mean)
print(thetime)
print(theloss)