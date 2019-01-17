import pickle
import os
import os.path
import random
from PIL import Image
import numpy as np

sampleRate = 8
celebDict = {}

with open("identity_CelebA.txt") as f:
    pairs = [x.split(" ") for x in f.readlines()]
    for pair in pairs:
        try:
            celebDict[pair[1]].append(pair[0].split(".")[0])
        except KeyError:
            celebDict[pair[1]] = [pair[0].split(".")[0]]

fPath = []
for k,v in celebDict.items():
    fPath += random.sample(v, len(v) // sampleRate)

#pathDict = {'train':["img_align_celeba/{}.jpg".format(x) for x in random.sample(fPath, len(fPath) - (len(fPath) // 10))], \
#            'test':["img_align_celeba/{}.jpg".format(x) for x in random.sample(fPath, len(fPath) // 10)]}
random.shuffle(fPath)
pathDict = {'train':["img_align_celeba/{}.jpg".format(x) for x in fPath[:len(fPath) - (len(fPath) // 10)]], \
            'test':["img_align_celeba/{}.jpg".format(x) for x in fPath[len(fPath) - (len(fPath) // 10):]]}
fullDSet = np.array([np.array(Image.open("img_align_celeba/{}.jpg".format(x))) for x in fPath])
pathDict['mean'] = fullDSet.mean((0, 1, 2)) / 255
pathDict['std'] = fullDSet.std((0, 1, 2)) / 255

pickle.dump(pathDict, open("img_align_celeba_flist.pkl", "wb+"))

print((pathDict['mean'], pathDict['std']))
