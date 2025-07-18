
import matplotlib
import matplotlib.pyplot as plt
import os.path as osp
from collections import defaultdict
import random
import cv2


data_root = '/data/mini-imagenet'

image_path = osp.join(data_root, 'images')
csv_path = osp.join(data_root, 'test.csv')
lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

wnids = []
lb = -1

data = []
targets = []
datatolable = dict()
class2data = defaultdict(list)

for l in lines:
    name, wnid = l.split(',')
    path = osp.join(image_path, name)
    if wnid not in wnids:
        wnids.append(wnid)
        lb += 1
    data.append(path)
    class2data[lb].append(path)
    targets.append(lb)
    datatolable[path] = wnid

# print(len(class2data[0]))
# 随机选取10个类 并挑选出相关的图像

classes = random.sample(wnids, 10)

# plt.figure()
# for i in range(1, 21, 2):
    
#     image_paths = random.sample(class2data[i // 2], 2)

#     img1 = cv2.imread(image_paths[0])
    
#     plt.subplot(2, 10, i)
#     plt.imshow(img1)
#     plt.title(image_paths[0])
#     plt.axis('off')

#     img2 = cv2.imread(image_paths[1])
#     plt.subplot(2, 10, i + 1)
#     plt.imshow(img2)
#     plt.title(image_paths[1])
#     plt.axis('off')

# plt.savefig("squares.png")

import scipy.stats as stats

pre=[30,31,34,40,36,35,
34,30,28,29]
post=[30,31,32,38,32,31,
32,29,28,30]
stats.ttest_rel(pre, post)