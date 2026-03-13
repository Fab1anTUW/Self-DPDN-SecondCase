# Self-Supervised Deep Prior Deformation Network

This repository is an adaption of the original Self-DPDN, to adapt a dataset so it works with SecondPose

Code for "Category-Level 6D Object Pose and Size Estimation using Self-Supervised Deep Prior Deformation Networks". ECCV2022.

[[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-20077-9_2) [[Arxiv]](https://arxiv.org/abs/2207.05444)

Created by [Jiehong Lin](https://jiehonglin.github.io/), Zewei Wei, Changxing Ding, and [Kui Jia](http://kuijia.site/).

## Requirements
The code has been tested with:

- python 3.10.19

The remaining packages are in the requirements.txt

## Data Processing

To make the Data Processing working, the luggage dataset has the be converted to the NOCS-dataset file structure like down below
```
data
└── NOCS
    ├── CAMERA
    │   ├── train
    │   └── val
    ├── camera_full_depths
    │   ├── train
    │   └── val
    ├── Real
    │   ├── train
    │   └── test
    ├── gts
    │   ├── val
    │   └── real_test
    ├── obj_models
    │   ├── train
    │   ├── val
    │   ├── real_train
    │   └── real_test
    ├── segmentation_results
    │   ├── train_trainedwoMask
    │   ├── test_trainedwoMask
    │   └── test_trainedwithMask
    └── mean_shapes.npy
```

The luggage dataset was provided as follows:
```
data
└── depth
├── mask
├── mask_visib
├── nocs
├── rgb
```

In the NOCS dataset a picture consists of RGB-, NOCS-, Depth-picture, mask, meta.txt and .pkl file.
The meta file has the following contents for synthetic data: <instance_id> <class_id> <scene_id> <model_name>
This meta file, as well as the bounding boxes, are created in the dataset conversion.
Because the masks from the luggage dataset are in separate files they need to be fused to one file.
Now all these pictures & files are sorted in to NOCS compatible scenes with 10 pictures (RGB, NOCS, Depth, Mask and Meta.txt) per folder.
Incompatible or defect pictures are skipped, with a summary at the end.
The converted dataset is stored under data/camera/train, to start the data_processing the folder data/camera/val has to be created. For processing just the training dataset this one can be empty.

All this conversion is done in the convert_to_nocs_dataset.py

After this conversion the normal Self-DPDN data_processing has to be used to prepare the data for SecondPose. Here the .pkl as well as the train_list.txt files are created.
According to the dataset the correct functions in main have to be selected/commmented

python data_processing.py

## Dataset conversion for training

To convert the luggage dataset it has to be stored in the test_data folder like this:

```
000000
└── depth
├── mask_visib
├── nocs
└── rgb
000001
└── depth
├── mask_visib
├── nocs
└── rgb
....
```

1. run convert_to_nocs_dataset.py
2. run data_processing.py (check if intrinsics in annotate_camera_train & annotate_test_data are set correctly)
3. copy the content from the data folder in the NOCS folder from SecondPose
4. run data_preprocess.py in SecondPose

## Dataset conversion for evaluation

1. run convert_to_nocs_dataset.py
2. run data_processing.py (check if intrinsics in annotate_camera_train & annotate_test_data are set correctly)
3. delete the empty val folder in data, copy train folder and rename to val
4. run data_processing.py again
5. run make_results_pkl_iterate.py
6. copy the content from the data folder in the NOCS folder from SecondPose
7. run data_preprocess.py in SecondPose (check if camera_test_stats = load_stats_test(.... is not commented)


## Acknowledgements

Our implementation leverages the code from [NOCS](https://github.com/hughw19/NOCS_CVPR2019), [DualPoseNet](https://github.com/Gorilla-Lab-SCUT/DualPoseNet), and [SPD](https://github.com/mentian/object-deformnet).


## Contact

