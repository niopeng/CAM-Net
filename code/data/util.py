import os
import pickle
import random
import numpy as np
import lmdb
import cv2
import json
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


####################
# Files & IO
####################


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def _get_paths_from_lmdb(dataroot):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    keys_cache_file = os.path.join(dataroot, '_keys_cache.p')
    if os.path.isfile(keys_cache_file):
        print('read lmdb keys from cache: {}'.format(keys_cache_file))
        keys = pickle.load(open(keys_cache_file, "rb"))
    else:
        with env.begin(write=False) as txn:
            print('creating lmdb keys cache: {}'.format(keys_cache_file))
            keys = [key.decode('ascii') for key, _ in txn.cursor()]
        pickle.dump(keys, open(keys_cache_file, 'wb'))
    paths = sorted([key for key in keys if not key.endswith('.meta')])
    return env, paths


def get_image_paths(data_type, dataroot):
    env, paths = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            env, paths = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return env, paths


def _read_lmdb_img(env, path):
    with env.begin(write=False) as txn:
        buf = txn.get(path.encode('ascii'))
        buf_meta = txn.get((path + '.meta').encode('ascii')).decode('ascii')
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    H, W, C = [int(s) for s in buf_meta.split(',')]
    img = img_flat.reshape(H, W, C)
    return img


def read_img(env, path):
    # reading images in RGB color space
    # read image by cv2 or from lmdb
    # return: Numpy float32, HWC, BGR, [0,1]
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        img = _read_lmdb_img(env, path)
    img_no_scale = img
    img = img.astype(np.float32) / 255.

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img_no_scale = np.expand_dims(img_no_scale, axis=2)

    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
        img_no_scale = img_no_scale[:, :, :3]

    return img, img_no_scale


class PaletteDataset(object):
    def __init__(self, info_path):
        with open(info_path, 'r') as fp:
            info = json.load(fp)
        self.palette = np.array(info['palette'], dtype=np.uint8)


def get_semantic_map(path, dataset):
    img = Image.open(path)
    img = np.asarray(img)  # HWC
    return _get_semantic_map(img, dataset)


def _get_semantic_map(semantic, dataset):
    """
    Load the label image and create the semantic map
    :param semantic: input segmented image
    :param dataset: json dataset
    :return: HWC (c = the number of objects in the color palette)
    """
    tmp = np.zeros((semantic.shape[0], semantic.shape[1], dataset.palette.shape[0]), dtype=np.float32)
    for k in range(dataset.palette.shape[0]):
        tmp[:, :, k] = np.float32((semantic[:, :, 0] == dataset.palette[k, 0]) &
                                  (semantic[:, :, 1] == dataset.palette[k, 1]) &
                                  (semantic[:, :, 2] == dataset.palette[k, 2]))

    tmp = np.concatenate((tmp, np.expand_dims(1 - np.sum(tmp, axis=2), axis=2)), axis=2)
    return tmp


####################
# image processing
# process on numpy image
####################
def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]
