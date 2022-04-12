# A loss function for change detection
**Accepted for publication at [IGARSS-22](https://www.igarss2022.org/default.php), Kuala Lumpur, Malaysia.**

Here, we provide the pytorch implementation of the paper: UAL: UNCHANGED AREA LOSS-FUNCTION FOR CHANGE DETECTION NETWORKS.

## Our Method

### Task Description

Given two images of the same scene acquired at different times, we are required to mark the changed 
and unchanged areas. Moreover, as for the changed areas, we need to annotate their detailed semantic masks. 

The change detection task in this competition can be decomposed into two sub-tasks:
* binary segmentation of changed and unchanged areas.
* semantic segmentation of changed areas.

### Model

![image](https://github.com/Chuan-shanjia/SenseEarth2020-ChangeDetection/blob/master/docs/schematic_%20diagram.png)

### My Improvement

In this project,we propose a loss function named UAL-function (Unchanged Area Loss-function). UAL aims to establish the semantic label correspondence within unchanged regions. It is simple and effective for improving semantic segmentation and change detection with respect to the feature separability. 

## Getting Started

### Dataset
[Description](https://rs.sensetime.com/competition/index.html#/data) | [Download [password: f3qq]](https://pan.baidu.com/s/1Yg90vlAiKezSoxH7WEoV6g) 

### Pretrained Model
[resnet-18](https://drive.google.com/file/d/1-vd9x7PMTgGTVQpAaNWF5tGy-NeLLdzB/view?usp=sharing) | [resnet-34](https://drive.google.com/file/d/1w68FmzmTTCRpLjS4pQGiR4IGGotL5iXo/view?usp=sharing) | [resnet-50](https://drive.google.com/file/d/1yvo8LT3rN4XhHR0nYfi2aVjtb1tL54mJ/view?usp=sharing)

### Final Trained Model
[fcn-resnet18](https://drive.google.com/file/d/1UfByShVuCxnsXVpCCFXAaYns_RYJ6rY9/view?usp=sharing) | [fcn-resnet34](https://drive.google.com/file/d/1NL80WmA3dzcoV3za-E0bvU0ZFOzjkLUL/view?usp=sharing) | [pspnet-resnet18](https://drive.google.com/file/d/1qsKBX4VU5RH_-yx4FXbRKXLprPPzSH7n/view?usp=sharing) | [pspnet-resnet34](https://drive.google.com/file/d/19Pdl1BR6_Hjdl9JcKTPwyR6dCYBb0C1G/view?usp=sharing)

### File Organization
```
# store the whole dataset and pretrained backbones
mkdir -p data/dataset ; mkdir -p data/pretrained_models ;

# store the trained models
mkdir -p outdir/models ; 

# store predictions of validation set and testing set
mkdir -p outdir/masks/val/im1 ; mkdir -p outdir/masks/val/im2 ;
mkdir -p outdir/masks/test/im1 ; mkdir -p outdir/masks/test/im2 ;

├── data
    ├── dataset                    # download from the link above
    │   ├── train                  # training set
    |   |   ├── im1
    |   |   └── ...
    │   └── val                    # the final testing set (without labels)
    |
    └── pretrained_models
        ├── resnet18.pth
        ├── resnet34.pth
        └── ...
```

### Training
```
# Please refer to utils/options.py for more arguments
# If hardware supports, more backbones can be trained, such as resnet50, resnet101
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --backbone "resnet18" --pretrained --model "fcn"
```

### Testing
```
# Modify the backbones, models and checkpoint paths in L39-44 in test.py manually according to your saved models
# Or simply use our final trained models
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py```

