import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import data.util as util


class LRHRFourLevelsDoubleDataset(data.Dataset):
    '''
    Read LR, HR and intermediate target image groups.
    If only HR image is provided, generate LR image on-the-fly.
    The group is matched by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LRHRFourLevelsDoubleDataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.paths_HR = None
        self.LR_env = None  # environment for lmdb
        self.HR_env = None

        # read image list from subset list txt
        if opt['subset_file'] is not None and opt['phase'] == 'train':
            with open(opt['subset_file']) as f:
                self.paths_HR = sorted([os.path.join(opt['dataroot_HR'], line.rstrip('\n')) for line in f])
            if opt['dataroot_LR'] is not None:
                raise NotImplementedError('Now subset only supports generating LR on-the-fly.')
        else:  # read image list from lmdb or image files
            self.HR_env, self.paths_HR = util.get_image_paths(opt['data_type'], opt['dataroot_HR'])
            self.LR_env, self.paths_LR = util.get_image_paths(opt['data_type'], opt['dataroot_LR'])
            self.LR_1_env, self.paths_LR_1 = util.get_image_paths(opt['data_type'], opt['dataroot_LR_1'])
            self.LR_2_env, self.paths_LR_2 = util.get_image_paths(opt['data_type'], opt['dataroot_LR_2'])
            self.LR_3_env, self.paths_LR_3 = util.get_image_paths(opt['data_type'], opt['dataroot_LR_3'])
            self.D1_env, self.paths_D1 = util.get_image_paths(opt['data_type'], opt['dataroot_D1'])
            self.D2_env, self.paths_D2 = util.get_image_paths(opt['data_type'], opt['dataroot_D2'])
            self.D3_env, self.paths_D3 = util.get_image_paths(opt['data_type'], opt['dataroot_D3'])

        assert self.paths_HR, 'Error: HR path is empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                'HR and LR datasets have different number of images - {}, {}.'.format( \
                    len(self.paths_LR), len(self.paths_HR))

        self.random_scale_list = [1]

    def __getitem__(self, index):
        HR_path, LR_path = None, None
        scale = self.opt['scale'] // 2

        # get HR image
        HR_path = self.paths_HR[index]
        img_HR, _ = util.read_img(self.HR_env, HR_path)

        LR_path = self.paths_LR[index]
        LR_1_path = self.paths_LR_1[index]
        LR_2_path = self.paths_LR_2[index]
        LR_3_path = self.paths_LR_3[index]
        D1_path = self.paths_D1[index]
        D2_path = self.paths_D2[index]
        D3_path = self.paths_D3[index]

        img_LR, _ = util.read_img(self.LR_env, LR_path)
        img_LR_1, _ = util.read_img(self.LR_1_env, LR_1_path)
        img_LR_2, _ = util.read_img(self.LR_2_env, LR_2_path)
        img_LR_3, _ = util.read_img(self.LR_3_env, LR_3_path)
        img_D1, _ = util.read_img(self.D1_env, D1_path)
        img_D2, _ = util.read_img(self.D2_env, D2_path)
        img_D3, _ = util.read_img(self.D3_env, D3_path)

        # if self.opt['phase'] == 'train':
        H, W, C = img_LR.shape
        HR_size = self.opt['HR_size'] if "HR_size" in self.opt else img_HR.shape[0]
        LR_size = HR_size // scale
        assert HR_size <= img_HR.shape[0] and HR_size <= img_HR.shape[1], "Target image too small for HR size"

        # randomly crop
        rnd_h = random.randint(0, max(0, H - LR_size))
        rnd_w = random.randint(0, max(0, W - LR_size))
        img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
        img_D1 = img_D1[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
        rnd_h *= 2
        rnd_w *= 2
        LR_size *= 2
        img_LR_1 = img_LR_1[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
        img_D2 = img_D2[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]

        rnd_h *= 2
        rnd_w *= 2
        LR_size *= 2
        img_LR_2 = img_LR_2[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
        img_D3 = img_D3[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]

        rnd_h *= 2
        rnd_w *= 2
        LR_size *= 2
        img_LR_3 = img_LR_3[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
        img_HR = img_HR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]

        # augmentation - flip, rotate
        img_LR, img_LR_1, img_LR_2, img_LR_3, img_HR, img_D1, img_D2, img_D3 = util.augment([img_LR,
                                                                                             img_LR_1,
                                                                                             img_LR_2,
                                                                                             img_LR_3,
                                                                                             img_HR,
                                                                                             img_D1,
                                                                                             img_D2,
                                                                                             img_D3],
                                                                                            self.opt['use_flip'],
                                                                                            self.opt['use_rot'])
        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
            img_LR_1 = img_LR_1[:, :, [2, 1, 0]]
            img_LR_2 = img_LR_2[:, :, [2, 1, 0]]
            img_LR_3 = img_LR_3[:, :, [2, 1, 0]]
            img_D1 = img_D1[:, :, [2, 1, 0]]
            img_D2 = img_D2[:, :, [2, 1, 0]]
            img_D3 = img_D3[:, :, [2, 1, 0]]
        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()
        img_LR_1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR_1, (2, 0, 1)))).float()
        img_LR_2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR_2, (2, 0, 1)))).float()
        img_LR_3 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR_3, (2, 0, 1)))).float()
        img_D1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_D1, (2, 0, 1)))).float()
        img_D2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_D2, (2, 0, 1)))).float()
        img_D3 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_D3, (2, 0, 1)))).float()

        if LR_path is None:
            LR_path = HR_path
        return {'network_input': [img_LR, img_LR_1, img_LR_2, img_LR_3],
                'HR': img_HR,
                'LR_path': LR_path,
                'HR_path': HR_path,
                'LR_1_path': LR_1_path,
                'LR_2_path': LR_2_path,
                'LR_3_path': LR_3_path,
                'D1': img_D1,
                'D2': img_D2,
                'D3': img_D3,
                'D1_path': D1_path,
                'D2_path': D2_path,
                'D3_path': D3_path,
                'is_valid': True}

    def __len__(self):
        return len(self.paths_HR)
