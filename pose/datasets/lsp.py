# Weichao Qiu, 2017
# TODO: Write a loader for virtual data
from __future__ import print_function, absolute_import

import torch.utils.data as data

import os

from scipy.io import loadmat

from pose.utils.imutils import load_image


class LSP(data.Dataset):
    def __init__(self, root_folder, train):
        if not os.path.isdir(root_folder):
            print('Folder not exist: %s' % root_folder)
            return

        self.image_path_template = os.path.join(root_folder, 'images/im%04d.jpg')

        mat_file = os.path.join(root_folder, 'joints.mat')
        gt_mat = loadmat(mat_file)
        joints = gt_mat['joints']
        # print(joints.shape) # 3, 14, 2000
        self.joints = joints
        # parse joints.mat
        self.train = train
        self.train_index = range(0,1000)
        self.test_index = range(1000,2000)

    def _global_index(self, index):
        if self.train:
            return self.train_index[index] # TODO: Handle out of bound error
        else:
            return self.test_index[index]

    def __getitem__(self, index):
        global_index = self._global_index(index)

        image_path = self.image_path_template % (global_index + 1)
        # Start from 1

        # Load image and convert the image to CxHxW
        im = load_image(image_path) # TODO: normalize etc.?
        x = im

        # Generate ground truth, copy from mpii.py
        label = self.joints[:,:,global_index]


        target = torch.zeros(nparts, self.out_res, self.out_res)
        # Create an empty heatmap
        for i in range(nparts):


        # Convert this ground truth to a heatmap
        return im, label

    def __len__(self):
        if self.train:
            return len(self.train_index)
        else:
            return len(self.test_index)

# For a dataset, it requires
# 1. Parse the ground truth file
# 2. Convert data point to training format
# 3. Provide a dataset specific function for visualization

if __name__ == '__main__':
    train = True
    train_dataset = LSP('lsp_dataset', train)
    train = False
    test_dataset = LSP('lsp_dataset', train)
    print(len(train_dataset))
    print(train_dataset[0])
    print(len(test_dataset))
    print(test_dataset[0])
