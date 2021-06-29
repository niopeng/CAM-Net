# CAM-Net: Cascading Modular Network for Multimodal Conditional Image Synthesis

[Project Page][project] | [Paper][paper] | [Pre-trained Models](/experiments/pretrained_models)

PyTorch implementation of CAM-Net: a unified architecture for multimodal conditional image synthesis.
CAM-Net is able to:

- (Colorization) Automatic colorizing a grayscale image
- (Super-Resolution) Increase the width and height of images by a factor of 16x
- (Image-Synthesis) Generating diverse images from semantic layouts
- (Decompression) Recover a plausible image from a heavily compressed image

![Alt Text](website/teaser.gif)

## Installation
Please refer to [this page](/code/).

## Organization
The repository consists of the following components:
- `code/`: Code for training and testing the model
- `experiments/`: Directory for checkpoints and plots
- `website/`: Resources for the project page
- `index.html`: Project page

[project]:https://niopeng.github.io/CAM-Net/
[paper]: https://arxiv.org/abs/2106.09015
[pretrain]: https://github.com/niopeng/CAM-Net/tree/main/experiments/pretrained_models
