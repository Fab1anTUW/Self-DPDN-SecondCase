# Self-Supervised Deep Prior Deformation Network

This repository is an adaptation  of the original Self-DPDN, to adapt a dataset so it works with SecondPose

## Requirements
The code has been tested with:

- python 3.10.19

The remaining packages are in the requirements.txt

## Data Processing

In order for the data processing to work, the luggage dataset must be converted to the NOCS dataset file structure shown below:
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
00000
└── depth
├── mask
├── mask_visib
├── nocs
└── rgb
```

In the NOCS dataset, a picture consists of an RGB picture, a NOCS picture, a depth picture, a mask, a meta.txt file and a .pkl file.
The meta file contains the following information for synthetic data: <instance_id> <class_id> <scene_id> <model_name>.
The meta file and the bounding boxes are created during dataset conversion.
As the masks from the luggage dataset are in separate files, they need to be combined into one file.
All these pictures and files are now sorted into NOCS-compatible scenes, with ten pictures (RGB, NOCS, depth, mask and meta.txt) per folder.
Incompatible or defective pictures are skipped, with a summary provided at the end.
The converted dataset is stored in the data/camera/train folder. To start data processing, the data/camera/val folder must be created. If you are only processing the training dataset, this folder can be empty.
All this conversion is done in the convert_to_nocs_dataset.py

After this conversion, the standard Self-DPDN data processing must be used to prepare the data for SecondPose. This is where the .pkl and train_list.txt files are created.
According to the dataset, the correct functions in the main program must be selected or commented.

Before running data_processing check if intrinsics in annotate_camera_train & annotate_test_data are set correctly, so the GTs are calculated accordingly
run data_processing.py

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
7. run data_preprocess.py in SecondPose (check if camera_test_stats = load_stats_test(.... is commented)


## Acknowledgements

Code for "Category-Level 6D Object Pose and Size Estimation using Self-Supervised Deep Prior Deformation Networks". ECCV2022.

[[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-20077-9_2) [[Arxiv]](https://arxiv.org/abs/2207.05444)

Created by [Jiehong Lin](https://jiehonglin.github.io/), Zewei Wei, Changxing Ding, and [Kui Jia](http://kuijia.site/).
