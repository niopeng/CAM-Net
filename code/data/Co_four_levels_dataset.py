import os.path
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util
from utils import util as img_util
import math


class CoFourLevelsDataset(data.Dataset):
    '''
    Supports up to 4 backbones.
    Generates D1, D2, D3 on-the-fly using PIL.
    HR_Color are the ground-truth color images in their original resolution; for example, 256 * 256, scale = 16
    D1 = color images, 32  * 32  = ((2/scale) * original)
    D2 = color images, 64  * 64  = ((4/scale) * original)
    D3 = color images, 128 * 128 = ((8/scale) * original)
    D4 = color images, 256 * 256 = HR_Color image

    In the colorization task, the input and output resolutions are equal; therefore, we wouldn't have LR_Color
    '''

    def __init__(self, opt):
        super(CoFourLevelsDataset, self).__init__()
        self.opt = opt
        self.paths_HR_Color = None
        self.HR_Color_env = None
        self.paths_D1 = None
        self.D1_env = None

        # read image list from subset list txt
        if opt['subset_file'] is not None and opt['phase'] == 'train':
            with open(opt['subset_file']) as f:
                self.paths_HR_Color = sorted([os.path.join(opt['dataroot_HR_Color'], line.rstrip('\n')) for line in f])
            if opt['dataroot_LR_Color'] is not None:
                raise NotImplementedError('Now subset only supports generating LR on-the-fly.')
        else:  # read image list from lmdb or image files
            self.HR_Color_env, self.paths_HR_Color = util.get_image_paths(opt['data_type'], opt['dataroot_HR_Color'])
            self.D1_env, self.paths_D1 = util.get_image_paths(opt['data_type'], opt['dataroot_D1'])
            self.D2_env, self.paths_D2 = util.get_image_paths(opt['data_type'], opt['dataroot_D2'])
            self.D3_env, self.paths_D3 = util.get_image_paths(opt['data_type'], opt['dataroot_D3'])

        assert self.paths_HR_Color, 'Error: HR_Color path is empty.'

        self.random_scale_list = [1]
        self.rarity_masks = []
        if opt['phase'] == 'train':
            if 'rarity_mask_1' in self.opt:
                for i in range(0, int(math.log(opt['scale'], 2))):
                    self.rarity_masks.append(np.load(opt['rarity_mask_{}'.format(i + 1)], mmap_mode='r'))

    def __getitem__(self, index):
        scale = self.opt['scale']

        validity = True
        # generating outputs for the specified IDs in the test set rather than generating outputs for the whole test set
        if self.opt['phase'] == 'val':
            if 'target_images_id' in self.opt:
                if self.paths_HR_Color[index] not in self.opt['target_images_id']:
                    return {'is_valid': False}

        # get HR_Color image
        HR_path = self.paths_HR_Color[index]
        img_HR_bgr, img_HR_bgr_no_scaled = util.read_img(self.HR_Color_env, HR_path)  # HWC, BGR, [0,1], [0, 255]

        # force to 3 channels
        if img_HR_bgr.ndim == 2:
            img_HR_bgr = cv2.cvtColor(img_HR_bgr, cv2.COLOR_GRAY2BGR)
            img_HR_bgr_no_scaled = cv2.cvtColor(img_HR_bgr_no_scaled, cv2.COLOR_GRAY2BGR)

        img_HR_rgb = cv2.cvtColor(img_HR_bgr, cv2.COLOR_BGR2RGB)  # HWC, RGB, [0, 1], 256 * 256
        img_HR_rgb_no_scaled = cv2.cvtColor(img_HR_bgr_no_scaled, cv2.COLOR_BGR2RGB)  # HWC, RGB, [0, 255], 256 * 256

        # D1, D2, D3, D4
        if self.paths_D1:
            D1_path = self.paths_D1[index]
            D2_path = self.paths_D2[index]
            D3_path = self.paths_D3[index]
            img_D1_bgr, _ = util.read_img(self.D1_env, D1_path)
            img_D2_bgr, _ = util.read_img(self.D2_env, D2_path)
            img_D3_bgr, _ = util.read_img(self.D3_env, D3_path)
            img_D1_rgb = cv2.cvtColor(img_D1_bgr, cv2.COLOR_BGR2RGB)
            img_D2_rgb = cv2.cvtColor(img_D2_bgr, cv2.COLOR_BGR2RGB)
            img_D3_rgb = cv2.cvtColor(img_D3_bgr, cv2.COLOR_BGR2RGB)
        else:  # down-sampling on-the-fly
            # HWC, RGB, [0, 1]
            img_D1_rgb = img_util.downsample_PIL(rgb_no_scaled=img_HR_rgb_no_scaled, scale=2.0 / scale)  # 32  * 32
            img_D2_rgb = img_util.downsample_PIL(rgb_no_scaled=img_HR_rgb_no_scaled, scale=4.0 / scale)  # 64  * 64
            img_D3_rgb = img_util.downsample_PIL(rgb_no_scaled=img_HR_rgb_no_scaled, scale=8.0 / scale)  # 128 * 128
            D1_path = D2_path = D3_path = HR_path

        # augmentation - flip, rotate
        if self.opt['phase'] == 'train':
            img_HR_rgb, img_D1_rgb, img_D2_rgb, img_D3_rgb = util.augment(
                [img_HR_rgb, img_D1_rgb, img_D2_rgb, img_D3_rgb],
                self.opt['use_flip'], self.opt['use_rot'])

        # L channel
        img_HR_lab = img_util.rgb2lab(img_HR_rgb)
        img_D1_lab = img_util.rgb2lab(img_D1_rgb)
        img_D2_lab = img_util.rgb2lab(img_D2_rgb)
        img_D3_lab = img_util.rgb2lab(img_D3_rgb)

        HR_L_channel = img_HR_lab[:, :, 0] / 100.0
        D1_L_channel = img_D1_lab[:, :, 0] / 100.0
        D2_L_channel = img_D2_lab[:, :, 0] / 100.0
        D3_L_channel = img_D3_lab[:, :, 0] / 100.0

        HR_L_channel_tensor = torch.Tensor(HR_L_channel)[None, :, :]
        D1_L_channel_tensor = torch.Tensor(D1_L_channel)[None, :, :]
        D2_L_channel_tensor = torch.Tensor(D2_L_channel)[None, :, :]
        D3_L_channel_tensor = torch.Tensor(D3_L_channel)[None, :, :]

        # HWC to CHW, numpy to tensor
        img_HR_tensor_rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR_rgb, (2, 0, 1)))).float()
        img_D1_tensor_rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(img_D1_rgb, (2, 0, 1)))).float()
        img_D2_tensor_rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(img_D2_rgb, (2, 0, 1)))).float()
        img_D3_tensor_rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(img_D3_rgb, (2, 0, 1)))).float()

        # rarity_masks
        if len(self.rarity_masks) != 0:
            rarity_mask1 = np.copy(self.rarity_masks[0][index])  # HWC
            rarity_mask2 = np.copy(self.rarity_masks[1][index])  # HWC
            rarity_mask3 = np.copy(self.rarity_masks[2][index])  # HWC
            rarity_mask4 = np.copy(self.rarity_masks[3][index])  # HWC

            rarity_mask1 = torch.from_numpy(np.ascontiguousarray(np.transpose(rarity_mask1, (2, 0, 1)))).float()
            rarity_mask2 = torch.from_numpy(np.ascontiguousarray(np.transpose(rarity_mask2, (2, 0, 1)))).float()
            rarity_mask3 = torch.from_numpy(np.ascontiguousarray(np.transpose(rarity_mask3, (2, 0, 1)))).float()
            rarity_mask4 = torch.from_numpy(np.ascontiguousarray(np.transpose(rarity_mask4, (2, 0, 1)))).float()

        if self.opt['phase'] == 'train':
            res = {
                'network_input': [D1_L_channel_tensor, D2_L_channel_tensor, D3_L_channel_tensor, HR_L_channel_tensor],
                'HR': img_HR_tensor_rgb,
                'HR_path': HR_path,
                'D1': img_D1_tensor_rgb,
                'D2': img_D2_tensor_rgb,
                'D3': img_D3_tensor_rgb,
                'D4': img_HR_tensor_rgb,
                'D1_path': D1_path,
                'D2_path': D2_path,
                'D3_path': D3_path,
                'D4_path': HR_path}
        else:
            res = {
                'network_input': [D1_L_channel_tensor, D2_L_channel_tensor, D3_L_channel_tensor, HR_L_channel_tensor],
                'HR': img_HR_tensor_rgb,
                'HR_path': HR_path,
                'is_valid': validity}

        if len(self.rarity_masks) != 0:
            res['rarity_masks'] = [rarity_mask1, rarity_mask2, rarity_mask3, rarity_mask4]
        return res

    def __len__(self):
        return len(self.paths_HR_Color)
