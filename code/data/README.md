# Dataloader

- read from **image** files OR from **.lmdb** for fast IO speed.
    - How to create .lmdb file? Please see [`create_lmdb_different_scales.py`](/code/data/create_lmdb_different_scales.py).

## Contents

- `LR_dataset`: only reads LR images in test phase where there is no target images.
- `LRHR_dataset`: reads LR and HR pairs from image folder or lmdb files. If only HR images are provided, downsample the images on-the-fly.
- `LRHR_four_levels_dataset`: similar to LRHR_dataset, include intermediate targets
- `LRHR_four_levels_double_dataset`: similar to LRHR_four_levels_dataset, include intermediate inputs
- `Co_four_levels_dataset`: reads color images and intermediate color targets (optional, the code can downsample automatically using PIL on-the-fly)
- `ImgSyn_five_levels_dataset`: reads images, labels, rarity masks, rarity bins, and the color pallet.
- `google_drive_downloader`: Downloads the pretrained models. Just replace the model's ID and the path you want to store it.

## Data Augmentation

We use random crop, random flip/rotation, (random scale) for data augmentation. 

## Data Preparation for Image Synthesis

Use [`generate_rarity_masks.py`](/code/data/generate_rarity_masks.py) to generate Rarity Bins (would be used to re-balance the dataset) and Rarity Masks (would be used in the loss calculation) in different resolutions.
