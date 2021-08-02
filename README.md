# Extreme Rotation Estimation using Dense Correlation Volumes

This repository contains a PyTorch implementation of the paper:

[*Extreme Rotation Estimation using Dense Correlation Volumes*](https://ruojincai.github.io/ExtremeRotation/)
[[Project page]](https://ruojincai.github.io/ExtremeRotation/)
[[Arxiv]](https://arxiv.org/abs/2104.13530)

[Ruojin Cai](http://www.cs.cornell.edu/~ruojin/), 
[Bharath Hariharan](http://home.bharathh.info/),
[Noah Snavely](http://www.cs.cornell.edu/~snavely/), 
[Hadar Averbuch-Elor](http://www.cs.cornell.edu/~hadarelor/)

CVPR 2021

## Introduction
We present a technique for estimating the relative 3D rotation of an RGB image pair in an extreme setting, where the images have little or no overlap. We observe that, even when images do not overlap, there may be rich hidden cues as to their geometric relationship, such as light source directions, vanishing points, and symmetries present in the scene. We propose a network design that can automatically learn such implicit cues by comparing all pairs of points between the two input images. Our method therefore constructs dense feature correlation volumes and processes these to predict relative 3D rotations. Our predictions are formed over a fine-grained discretization of rotations, bypassing difficulties associated with regressing 3D rotations. We demonstrate our approach on a large variety of extreme RGB image pairs, including indoor and outdoor images captured under different lighting conditions and geographic locations. Our evaluation shows that our model can successfully estimate relative rotations among non-overlapping images without compromising performance over overlapping image pairs.

#### Overview of our Method:
![Overview](https://ruojincai.github.io/ExtremeRotation/assets/overview.png)

Given a pair of images, a shared-weight Siamese encoder extracts feature maps. We compute a 4D correlation volume using the inner product of features, from which our model predicts the relative rotation (here, as distributions over Euler angles).


## Dependencies
```bash
# Create conda environment with python 3.6, torch 1.3.1 and CUDA 10.0
conda env create -f ./tools/environment.yml
conda activate rota
```

## Dataset

Perspective images are randomly sampled from panoramas with a resolution of 256 × 256 and a 90◦ FoV. 
We sample images distributed uniformly over the range of [−180, 180] for yaw angles.
To avoid generating textureless images that focus on the ceiling/sky or the floor, we limit the range over pitch angles to [−30◦, 30◦] for the indoor datasets and [−45◦, 45◦] for the outdoor dataset.

Download [InteriorNet](https://interiornet.org/), [SUN360](https://vision.cs.princeton.edu/projects/2012/SUN360/data/), and [StreetLearn](https://sites.google.com/view/streetlearn/dataset) datasets to obtain the full panoramas.

Metadata files about the training and test image pairs are available in the following google drive: [link](https://drive.google.com/drive/folders/1xA6O-FYAKWj0Ed2E3qIu-tKnw29C9q1Z?usp=sharing).
Download the `metadata.zip` file, unzip it and put it under the project root directory.

We base on this MATLAB [Toolbox](https://github.com/yindaz/PanoBasic) that extracts perspective images from an input panorama.
Before running `PanoBasic/pano2perspective_script.m`, you need to modify the path to the datasets and metadata files in the script.

## Pretrained Model 

Pretrained models are be available in the following google drive: [link](https://drive.google.com/drive/folders/1xA6O-FYAKWj0Ed2E3qIu-tKnw29C9q1Z?usp=sharing).
To use the pretrained models, download the `pretrained.zip` file, unzip it and put it under the project root directory.

#### Testing the pretrained model:
The following commands test the performance of the pre-trained models in the rotation estimation task.
The commands output the mean and median geodesic error, and the percentage of pairs with a relative rotation error under 10◦ for different levels of overlap on the test set.
```bash
# Usage:
# python test.py <config> --pretrained <checkpoint_filename>

python test.py configs/sun360/sun360_cv_distribution.yaml \
    --pretrained pretrained/sun360_cv_distribution.pt

python test.py configs/interiornet/interiornet_cv_distribution.yaml \
    --pretrained pretrained/interiornet_cv_distribution.pt

python test.py configs/interiornetT/interiornetT_cv_distribution.yaml \
    --pretrained pretrained/interiornetT_cv_distribution.pt

python test.py configs/streetlearn/streetlearn_cv_distribution.yaml \
    --pretrained pretrained/streetlearn_cv_distribution.pt

python test.py configs/streetlearnT/streetlearnT_cv_distribution.yaml \
    --pretrained pretrained/streetlearnT_cv_distribution.pt
```

Rotation estimation evaluation of the pretrained models is as follows:
|       |        | InteriorNet |        |   |        | InteriorNet-T |        |   |        | SUM360 |        |   |        | StreetLearn |        |   |        | StreetLearn-T |        |
|-------|:------:|:-----------:|:------:|---|:------:|:-------------:|:------:|---|:------:|:------:|:------:|---|:------:|:-----------:|:------:|---|:------:|:-------------:|:------:|
|       | Avg(°) | Med(°)      | 10°    |   | Avg(°) | Med(°)        | 10°    |   | Avg(°) | Med(°) | 10°    |   | Avg(°) | Med(°)      | 10°    |   | Avg(°) | Med(°)        | 10°    |
| Large |  1.82  |     0.88    | 98.76% |   |  8.86  |      1.86     | 93.13% |   |  1.37  |  1.09  | 99.51% |   |  1.38  |     1.12    | 100.00%|   |  24.98 |      2.50     | 78.95% |
| Small |  4.31  |     1.16    | 96.58% |   |  30.43 |      2.63     | 74.07% |   |  6.13  |  1.77  | 95.86% |   |  3.25  |     1.41    | 98.34% |   |  27.84 |      3.19     | 74.76% |
| None  |  37.69 |     3.15    | 61.97% |   |  49.44 |      4.17     | 58.36% |   |  34.92 |  4.43  | 61.39% |   |  5.46  |     1.65    | 96.60% |   |  32.43 |      3.64     | 72.69% |
| All   |  13.49 |     1.18    | 86.90% |   |  29.68 |      2.58     | 75.10% |   |  20.45 |  2.23  | 78.30% |   |  4.10  |     1.46    | 97.70% |   |  29.85 |      3.19     | 74.30% |


## Training

```bash
# Usage:
# python train.py <config>

python train.py configs/interiornet/interiornet_cv_distribution.yaml

python train.py configs/interiornetT/interiornetT_cv_distribution.yaml

python train.py configs/sun360/sun360_cv_distribution_overlap.yaml
python train.py configs/sun360/sun360_cv_distribution.yaml --resume --pretrained <checkpoint_filename>

python train.py configs/streetlearn/streetlearn_cv_distribution_overlap.yaml
python train.py configs/streetlearn/streetlearn_cv_distribution.yaml --resume --pretrained <checkpoint_filename>

python train.py configs/streetlearnT/streetlearnT_cv_distribution_overlap.yaml
python train.py configs/streetlearnT/streetlearnT_cv_distribution.yaml --resume --pretrained <checkpoint_filename>
```

For SUN360 and StreetLearn dataset, finetune from the pretrained model, which is training with only overlapping pairs, at epoch 10.
More configs about baselines can be found in the folder `configs/sun360`.

# Cite 
Please cite our work if you find it useful: 
```bibtex
@inproceedings{Cai2021Extreme,
 title={Extreme Rotation Estimation using Dense Correlation Volumes},
 author={Cai, Ruojin and Hariharan, Bharath and Snavely, Noah and Averbuch-Elor, Hadar},
 booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
 year={2021}
}
```

#### Acknowledgment
This work was supported in part by the National Science Foundation (IIS-2008313) and by the generosity of Eric and Wendy Schmidt by recommendation of the Schmidt Futures program and the Zuckerman STEM leadership program.

