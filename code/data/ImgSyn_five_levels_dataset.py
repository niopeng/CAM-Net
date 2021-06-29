import os.path
import numpy as np
import torch
import torch.utils.data as data
import data.util as util
import math


class ImageLabelDatasetFiveLevels(data.Dataset):
    def __init__(self, opt):
        super(ImageLabelDatasetFiveLevels, self).__init__()
        self.opt = opt

        # read image list from subset list txt
        if opt['subset_file'] is not None and opt['phase'] == 'train':
            with open(opt['subset_file']) as f:
                self.paths_Images = sorted([os.path.join(opt['dataroot_HR_Image'], line.rstrip('\n')) for line in f])
            if opt['dataroot_LR_Image'] is not None:
                raise NotImplementedError('Now subset only supports generating LR on-the-fly.')
        else:  # read image list from lmdb or image files
            self.HR_images_env, self.paths_HR_images = util.get_image_paths(opt['data_type'], opt['dataroot_HR_Image'])

            self.LR_images_env, self.paths_LR_images = util.get_image_paths(opt['data_type'], opt['dataroot_LR_Image'])
            self.LR_labels_env, self.paths_LR_labels = util.get_image_paths(opt['data_type'], opt['dataroot_LR_Label'])

            self.D1_images_env, self.paths_D1_images = util.get_image_paths(opt['data_type'], opt['dataroot_D1_Image'])
            self.D1_labels_env, self.paths_D1_labels = util.get_image_paths(opt['data_type'], opt['dataroot_D1_Label'])

            self.D2_images_env, self.paths_D2_images = util.get_image_paths(opt['data_type'], opt['dataroot_D2_Image'])
            self.D2_labels_env, self.paths_D2_labels = util.get_image_paths(opt['data_type'], opt['dataroot_D2_Label'])

            self.D3_images_env, self.paths_D3_images = util.get_image_paths(opt['data_type'], opt['dataroot_D3_Image'])
            self.D3_labels_env, self.paths_D3_labels = util.get_image_paths(opt['data_type'], opt['dataroot_D3_Label'])

            self.D4_images_env, self.paths_D4_images = util.get_image_paths(opt['data_type'], opt['dataroot_D4_Image'])
            self.D4_labels_env, self.paths_D4_labels = util.get_image_paths(opt['data_type'], opt['dataroot_D4_Label'])

            print("Images: {}".format(len(self.paths_HR_images)))
            print("Labels: {}".format(len(self.paths_LR_labels)))

        assert self.paths_HR_images, 'Error: Images path is empty.'
        if self.paths_HR_images and self.paths_LR_images:
            assert len(self.paths_HR_images) == len(self.paths_LR_images), \
                'Images and Labels datasets have different number of images - {}, {}.'.format( \
                    len(self.paths_HR_images), len(self.paths_LR_images))

        self.random_scale_list = [1]
        # rarity maks
        self.rarity_masks = []

        if opt['phase'] == 'train':
            # batch size for modifying the input index in the training time

            self.batch_size = opt["batch_size_per_month"]
            if 'rarity_mask_1' in self.opt:
                for i in range(0, int(math.log(opt['scale'], 2))):
                    self.rarity_masks.append(np.load(opt['rarity_mask_{}'.format(i + 1)], mmap_mode='r'))

            # rarity bins for dataset re-balancing
            self.objectid = [0, 2, 10, 19, 1]
            self.objectnum = 5
            self.rarity_bin = []
            for i in range(self.objectnum):
                self.rarity_bin.append(np.load(opt['rarity_bins'] % self.objectid[i]))

        # color palette dataset
        self.dataset_color_map = util.PaletteDataset(opt['palette'])

    def get_modified_index(self, inp_idx):
        inp_idx = inp_idx % self.batch_size
        idx = np.searchsorted(self.rarity_bin[inp_idx % self.objectnum], np.random.rand())
        return idx

    def __getitem__(self, index):
        """
        In Progressive Image Synthesis:
        HR = the highest ground-truth resolution image; for example: 256 * 512

        Scale = 32
        LR = 8   * 16  = ((1/scale)   * original)
        D1 = 16  * 32  = ((2/scale)   * original)
        D2 = 32  * 64  = ((4/scale)   * original)
        D3 = 64  * 128 = ((8/scale)   * original)
        D4 = 128 * 256 = ((16/scale)  * original)
        HR = 256 * 512 = ((64/scale)  * original)

        """
        if self.opt['phase'] == 'train':
            index = self.get_modified_index(index)
            if index == 9061:  # this image is blank
                index += 1

        validity = True
        if self.opt['phase'] == 'val':
            if 'target_images_id' in self.opt:
                if self.paths_HR_images[index] not in self.opt['target_images_id']:
                    return {'is_valid': False}

        # get images
        HR_image_path = self.paths_HR_images[index]
        LR_image_path = self.paths_LR_images[index]
        D1_image_path = self.paths_D1_images[index]
        D2_image_path = self.paths_D2_images[index]
        D3_image_path = self.paths_D3_images[index]
        D4_image_path = self.paths_D4_images[index]

        # 1: Load Images
        _, HR_image_bgr_no_scaled = util.read_img(self.HR_images_env, HR_image_path)  # BGR, HWC, [0, 255]
        HR_image = HR_image_bgr_no_scaled[:, :, [2, 1, 0]]  # RGB, HWC, [0, 255]

        _, D1_image_bgr_no_scaled = util.read_img(self.D1_images_env, D1_image_path)
        D1_image = D1_image_bgr_no_scaled[:, :, [2, 1, 0]]  # RGB, HWC, [0, 255]

        _, D2_image_bgr_no_scaled = util.read_img(self.D2_images_env, D2_image_path)
        D2_image = D2_image_bgr_no_scaled[:, :, [2, 1, 0]]  # RGB, HWC, [0, 255]

        _, D3_image_bgr_no_scaled = util.read_img(self.D3_images_env, D3_image_path)
        D3_image = D3_image_bgr_no_scaled[:, :, [2, 1, 0]]  # RGB, HWC, [0, 255]

        _, D4_image_bgr_no_scaled = util.read_img(self.D4_images_env, D4_image_path)
        D4_image = D4_image_bgr_no_scaled[:, :, [2, 1, 0]]  # RGB, HWC, [0, 255]

        # 2: HWC to CHW, numpy to tensor, [0, 255] - > [0, 1]
        HR_image = torch.from_numpy(
            np.ascontiguousarray(np.transpose(HR_image.astype(np.float16) / 255., (2, 0, 1)))).float()
        D1_image = torch.from_numpy(
            np.ascontiguousarray(np.transpose(D1_image.astype(np.float16) / 255., (2, 0, 1)))).float()
        D2_image = torch.from_numpy(
            np.ascontiguousarray(np.transpose(D2_image.astype(np.float16) / 255., (2, 0, 1)))).float()
        D3_image = torch.from_numpy(
            np.ascontiguousarray(np.transpose(D3_image.astype(np.float16) / 255., (2, 0, 1)))).float()
        D4_image = torch.from_numpy(
            np.ascontiguousarray(np.transpose(D4_image.astype(np.float16) / 255., (2, 0, 1)))).float()

        # get labels
        LR_label_path = self.paths_LR_labels[index]
        D1_label_path = self.paths_D1_labels[index]
        D2_label_path = self.paths_D2_labels[index]
        D3_label_path = self.paths_D3_labels[index]
        D4_label_path = self.paths_D4_labels[index]

        # 1: Load Labels
        _, LR_label_bgr_no_scaled = util.read_img(self.LR_labels_env, LR_label_path)  # BGR, HWC, [0, 255]
        LR_label = LR_label_bgr_no_scaled[:, :, [2, 1, 0]]  # RGB, HWC, [0, 255]

        _, D1_label_bgr_no_scaled = util.read_img(self.D1_labels_env, D1_label_path)  # BGR, HWC, [0, 255]
        D1_label = D1_label_bgr_no_scaled[:, :, [2, 1, 0]]  # RGB, HWC, [0, 255]

        _, D2_label_bgr_no_scaled = util.read_img(self.D2_labels_env, D2_label_path)  # BGR, HWC, [0, 255]
        D2_label = D2_label_bgr_no_scaled[:, :, [2, 1, 0]]  # RGB, HWC, [0, 255]

        _, D3_label_bgr_no_scaled = util.read_img(self.D3_labels_env, D3_label_path)  # BGR, HWC, [0, 255]
        D3_label = D3_label_bgr_no_scaled[:, :, [2, 1, 0]]  # RGB, HWC, [0, 255]

        _, D4_label_bgr_no_scaled = util.read_img(self.D4_labels_env, D4_label_path)  # BGR, HWC, [0, 255]
        D4_label = D4_label_bgr_no_scaled[:, :, [2, 1, 0]]  # RGB, HWC, [0, 255]

        # 2: get semantic maps
        LR_label = util._get_semantic_map(LR_label, self.dataset_color_map)  # HWC, One-Hot Encoded
        D1_label = util._get_semantic_map(D1_label, self.dataset_color_map)  # HWC, One-Hot Encoded
        D2_label = util._get_semantic_map(D2_label, self.dataset_color_map)  # HWC, One-Hot Encoded
        D3_label = util._get_semantic_map(D3_label, self.dataset_color_map)  # HWC, One-Hot Encoded
        D4_label = util._get_semantic_map(D4_label, self.dataset_color_map)  # HWC, One-Hot Encoded

        # 3: HWC to CHW, numpy to tensor
        LR_label = torch.from_numpy(np.ascontiguousarray(np.transpose(LR_label, (2, 0, 1)))).float()
        D1_label = torch.from_numpy(np.ascontiguousarray(np.transpose(D1_label, (2, 0, 1)))).float()
        D2_label = torch.from_numpy(np.ascontiguousarray(np.transpose(D2_label, (2, 0, 1)))).float()
        D3_label = torch.from_numpy(np.ascontiguousarray(np.transpose(D3_label, (2, 0, 1)))).float()
        D4_label = torch.from_numpy(np.ascontiguousarray(np.transpose(D4_label, (2, 0, 1)))).float()

        # rarity masks
        if len(self.rarity_masks) != 0:
            rarity_mask1 = np.copy(self.rarity_masks[0][index])  # HWC
            rarity_mask2 = np.copy(self.rarity_masks[1][index])  # HWC
            rarity_mask3 = np.copy(self.rarity_masks[2][index])  # HWC
            rarity_mask4 = np.copy(self.rarity_masks[3][index])  # HWC
            rarity_mask5 = np.copy(self.rarity_masks[4][index])  # HWC

            rarity_mask1 = torch.from_numpy(np.ascontiguousarray(np.transpose(rarity_mask1, (2, 0, 1)))).float()
            rarity_mask2 = torch.from_numpy(np.ascontiguousarray(np.transpose(rarity_mask2, (2, 0, 1)))).float()
            rarity_mask3 = torch.from_numpy(np.ascontiguousarray(np.transpose(rarity_mask3, (2, 0, 1)))).float()
            rarity_mask4 = torch.from_numpy(np.ascontiguousarray(np.transpose(rarity_mask4, (2, 0, 1)))).float()
            rarity_mask5 = torch.from_numpy(np.ascontiguousarray(np.transpose(rarity_mask5, (2, 0, 1)))).float()

        res = {'network_input': [LR_label, D1_label, D2_label, D3_label, D4_label],
               'HR': HR_image,
               'LR_path': LR_image_path,
               'HR_path': HR_image_path,
               'D1': D1_image,
               'D2': D2_image,
               'D3': D3_image,
               'D4': D4_image,
               'D1_path': D1_image_path,
               'D2_path': D2_image_path,
               'D3_path': D3_image_path,
               'D4_path': D4_image_path,
               'is_valid': validity}
        if len(self.rarity_masks) != 0:
            res['rarity_masks'] = [rarity_mask1, rarity_mask2, rarity_mask3, rarity_mask4, rarity_mask5]
        return res

    def __len__(self):
        return len(self.paths_HR_images)
