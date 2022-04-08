


from __future__ import print_function
import argparse

import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--numSamplesToDisp', type=int, default=1, help='model index')
parser.add_argument('--dataset', type=str, default='', help='dataset path')
parser.add_argument('--class_choice', type=str, default='', help='class choice')

opt = parser.parse_args()
print(opt)
state_dict = torch.load(opt.model)
classifier = PointNetDenseCls(k= state_dict['conv4.weight'].size()[0])
classifier.load_state_dict(state_dict)
classifier.eval()

d = ShapeNetDataset(
    root=opt.dataset,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)

cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]

for idx in range(opt.numSamplesToDisp):
  point, seg = d[idx]
  point_np = point.numpy()
  point = point.transpose(1, 0).contiguous()
  gt = cmap[seg.numpy() - 1, :]
  point = Variable(point.view(1, point.size()[0], point.size()[1]))
  pred, _, _ = classifier(point)
  pred_choice = pred.data.max(2)[1]
  point_cloud = pv.PolyData(point_np)
  
  pred_color = cmap[pred_choice.numpy()[0], :]
  point_cloud['colors'] = pred_color
  #print(pred_color[1,:])
  #print(pred_color.shape)  

  point_cloud.plot(notebook=True, window_size=(600,400))

#print(pred_color.shape)


