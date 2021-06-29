import numpy as np
from scipy import stats
from util import PaletteDataset
from PIL import Image


def kdeforvoid(a):
    return np.ones(a.shape[1])


def get_image(path):
    return np.asarray(Image.open(path))  # HWC


def get_semantic_map(semantic, dataset):
    tmp = np.zeros((semantic.shape[0], semantic.shape[1], dataset.palette.shape[0]), dtype=np.float32)
    for k in range(dataset.palette.shape[0]):
        tmp[:, :, k] = np.float32(
            (semantic[:, :, 0] == dataset.palette[k, 0]) & (semantic[:, :, 1] == dataset.palette[k, 1]) & (
                    semantic[:, :, 2] == dataset.palette[k, 2]))
    return tmp.reshape((1,) + tmp.shape)  # (1, 256, 512, 19) -> one hot coding


original_h = 256
original_w = 512
scales = [1., 1 / 2., 1 / 4., 1 / 8., 1 / 16., 1 / 32.]
root_original_size = '/path/to/data/preprocessed/'  # variable
root_save = '/path/to/save/Rarity_Masks_Bins/'  # variable
dataset_color_map = PaletteDataset('cityscapes.json')

for sc in scales:
    print("#####################################")
    print("scale: {}".format(sc))
    label_root = root_original_size + "{}x{}/labels/train/".format(int(original_h * sc), int(original_w * sc))
    image_root = root_original_size + "{}x{}/images/train/".format(int(original_h * sc), int(original_w * sc))

    num_img = 7000  # train
    avgcolor = np.empty([num_img, 20, 3])
    nums = np.zeros(20, dtype=np.int)
    areas = np.empty([num_img, 20])

    for i in range(num_img):
        semantic = get_semantic_map(get_image(label_root + "%08d.png" % (i + 1)), dataset_color_map)  # variable
        semantic = np.concatenate((semantic, np.expand_dims(1 - np.sum(semantic, axis=3), axis=3)), axis=3)
        image = get_image(image_root + "%08d.png" % (i + 1))  # variable
        areas[i] = np.sum(semantic, axis=(0, 1, 2))
        avgcolor[i] = np.sum(np.multiply(np.transpose(semantic, (3, 1, 2, 0)), image), axis=(1, 2)) / np.expand_dims(
            areas[i], 1)

    kernels = []
    invalidid = []

    for i in range(20):
        base = avgcolor[:, i, :][~np.any(np.isnan(avgcolor[:, i, :]), axis=1)]
        if base.shape[0] <= 67:
            print("skip {}".format(i))
            kernels.append(None)
            invalidid.append(i)
            continue
        values = np.transpose(base)
        kernels.append(stats.gaussian_kde(values))
        print("{}, {}".format(i, base.shape))

    rarity = np.zeros([num_img, 20], dtype=np.float64)
    clusterres = np.zeros((num_img, 20), dtype=np.int)
    rarity_mask = np.empty([num_img, int(original_h * sc), int(original_w * sc), 1], dtype=np.float32)
    objectlist = ['road', 'building', 'vegetation', 'other', 'car', 'sidewalk']
    objectid = range(20)  # +[100]

    for i in range(num_img):
        maxscore = 0
        semantic = get_semantic_map(get_image(label_root + "%08d.png" % (i + 1)), dataset_color_map)  # variable
        semantic = np.concatenate((semantic, np.expand_dims(1 - np.sum(semantic, axis=3), axis=3)), axis=3)
        scores = np.zeros([20], dtype=np.float32)
        for objid in range(20):
            if np.isnan(avgcolor[i, objid, 0]):
                continue
            else:
                if objid in invalidid:
                    prob = maxscore
                else:
                    prob = kernels[objid](avgcolor[i, objid])
                rarity[i, objid] += 1. / prob
                scores[objid] = 1. / prob
                maxscore = max(maxscore, scores[objid])

        rarity_mask[i] = np.expand_dims(np.sum(np.multiply(semantic, scores), axis=(0, 3)), 2) / maxscore

    save_path = root_save + "GTA_weighted_rarity_mask_{}x{}.npy".format(int(original_h * sc), int(original_w * sc))
    np.save(save_path, rarity_mask)

    if sc == 1:
        for objid in objectid:
            objname = str(objid)
            rarity_bin = rarity[:, objid] / np.sum(rarity[:, objid])
            for i in range(1, num_img):
                rarity_bin[i] += rarity_bin[i - 1]
            save_temp = root_save + "kdecolor_rarity_bin_{}.npy".format(objid)
            np.save(save_temp, rarity_bin)
        print("scale is {}, rarity bins are generated...".format(sc))

print("Finish...")
