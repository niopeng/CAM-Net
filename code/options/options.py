import os
from collections import OrderedDict
from datetime import datetime
import json


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def parse(opt_path, is_train=True):
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    opt['timestamp'] = get_timestamp()
    opt['is_train'] = is_train
    scale = opt['scale']

    # datasets
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['scale'] = scale

        HR_key_2 = None
        LR_key_2 = None
        HR_bg_Key_2 = None
        if opt['task'] == "Colorization":
            HR_key = "dataroot_HR_Color"
            LR_key = "dataroot_LR_Color"
            HR_bg_Key = "dataroot_HR_Color_bg"
        elif opt['task'] == "Super_Resolution" or opt['task'] == "Decompression":
            HR_key = "dataroot_HR"
            LR_key = "dataroot_LR"
            HR_bg_Key = "dataroot_HR_bg"
        elif opt['task'] == "Image_Synthesis":
            HR_key = "dataroot_HR_Image"
            HR_key_2 = "dataroot_HR_Label"
            HR_bg_Key = "dataroot_HR_Image_bg"
            HR_bg_Key_2 = "dataroot_HR_Label_bg"
            LR_key = "dataroot_LR_Image"
            LR_key_2 = "dataroot_LR_Label"

        is_lmdb = False
        if HR_key in dataset and dataset[HR_key] is not None:
            dataset[HR_key] = os.path.expanduser(dataset[HR_key])
            if HR_key_2 is not None:
                dataset[HR_key_2] = os.path.expanduser(dataset[HR_key_2])
            if dataset[HR_key].endswith('lmdb'):
                is_lmdb = True
        if HR_bg_Key in dataset and dataset[HR_bg_Key] is not None:
            dataset[HR_bg_Key] = os.path.expanduser(dataset[HR_bg_Key])
            if HR_bg_Key_2 is not None:
                dataset[HR_bg_Key_2] = os.path.expanduser(dataset[HR_bg_Key_2])
        if LR_key in dataset and dataset[LR_key] is not None:
            dataset[LR_key] = os.path.expanduser(dataset[LR_key])
            if LR_key_2 is not None:
                dataset[LR_key_2] = os.path.expanduser(dataset[LR_key_2])
            if dataset[LR_key].endswith('lmdb'):
                is_lmdb = True
        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'

        if phase == 'train' and 'subset_file' in dataset and dataset['subset_file'] is not None:
            dataset['subset_file'] = os.path.expanduser(dataset['subset_file'])

    # path
    for key, path in opt['path'].items():
        if path and key in opt['path']:
            opt['path'][key] = os.path.expanduser(path)
    if is_train:
        experiments_root = os.path.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = os.path.join(experiments_root, 'models')
        opt['path']['log'] = experiments_root
        opt['path']['val_images'] = os.path.join(experiments_root, 'val_images')

        # change some options for debug mode
        if 'debug' in opt['name']:
            opt['train']['val_freq'] = 8
            opt['logger']['print_freq'] = 2
            opt['logger']['save_checkpoint_freq'] = 8
            opt['train']['lr_decay_iter'] = 10
    else:  # test
        results_root = os.path.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root

    # network
    opt['network_G']['scale'] = scale

    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    return opt


def save(opt):
    dump_dir = opt['path']['experiments_root'] if opt['is_train'] else opt['path']['results_root']
    dump_path = os.path.join(dump_dir, 'options.json')
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
