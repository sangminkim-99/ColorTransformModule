# ColorTransformModule
Official Implementation of PG2023 Paper - Robust View Synthesis with Color Transform Module ([project page](#), [paper](https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.14931))


## Overview

In this repository, you can test the Color Transform Module applied on [DirectVoxGO](https://github.com/sunset1995/DirectVoxGO).


## Directory structure for the datasets

<details>
  <summary> (click to expand;) </summary>

    data
    ├── nerf_synthetic     # Link: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
    │   └── [chair|drums|ficus|hotdog|lego|materials|mic|ship]
    │       ├── [train|val|test]
    │       │   └── r_*.png
    │       └── transforms_[train|val|test].json
    │
    ├── Synthetic_NSVF     # Link: https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip
    │   └── [Bike|Lifestyle|Palace|Robot|Spaceship|Steamtrain|Toad|Wineholder]
    │       ├── intrinsics.txt
    │       ├── rgb
    │       │   └── [0_train|1_val|2_test]_*.png
    │       └── pose
    │           └── [0_train|1_val|2_test]_*.txt
    │
    ├── BlendedMVS         # Link: https://dl.fbaipublicfiles.com/nsvf/dataset/BlendedMVS.zip
    │   └── [Character|Fountain|Jade|Statues]
    │       ├── intrinsics.txt
    │       ├── rgb
    │       │   └── [0|1|2]_*.png
    │       └── pose
    │           └── [0|1|2]_*.txt
    │
    ├── TanksAndTemple     # Link: https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip
    │   └── [Barn|Caterpillar|Family|Ignatius|Truck]
    │       ├── intrinsics.txt
    │       ├── rgb
    │       │   └── [0|1|2]_*.png
    │       └── pose
    │           └── [0|1|2]_*.txt
    │
    ├── deepvoxels         # Link: https://drive.google.com/drive/folders/1ScsRlnzy9Bd_n-xw83SP-0t548v63mPH
    │   └── [train|validation|test]
    │       └── [armchair|cube|greek|vase]
    │           ├── intrinsics.txt
    │           ├── rgb/*.png
    │           └── pose/*.txt
    │
    ├── nerf_llff_data     # Link: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
    │   └── [fern|flower|fortress|horns|leaves|orchids|room|trex]
    │
    ├── tanks_and_temples  # Link: https://drive.google.com/file/d/11KRfN91W1AxAW6lOFs4EeYDbeoQZCi87/view?usp=sharing
    │   └── [tat_intermediate_M60|tat_intermediate_Playground|tat_intermediate_Train|tat_training_Truck]
    │       └── [train|test]
    │           ├── intrinsics/*txt
    │           ├── pose/*txt
    │           └── rgb/*jpg
    │
    ├── lf_data            # Link: https://drive.google.com/file/d/1gsjDjkbTh4GAR9fFqlIDZ__qR9NYTURQ/view?usp=sharing
    │   └── [africa|basket|ship|statue|torch]
    │       └── [train|test]
    │           ├── intrinsics/*txt
    │           ├── pose/*txt
    │           └── rgb/*jpg
    │
    ├── 360_v2             # Link: https://jonbarron.info/mipnerf360/
    │   └── [bicycle|bonsai|counter|garden|kitchen|room|stump]
    │       ├── poses_bounds.npy
    │       └── [images_2|images_4]
    │
    ├── nerf_llff_data     # Link: https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7
    │   └── [fern|flower|fortress|horns|leaves|orchids|room|trex]
    │       ├── poses_bounds.npy
    │       └── [images_2|images_4]
    │
    ├── rawnerf     # Link: http://storage.googleapis.com/gresearch/refraw360/raw.zip
    │   └── scenes
    │       └── [bikes|candlefiat|livingroom|morningkitchen|nightstreet|notchbush|parkstatue|scooter|streetcorner]
    │           ├── poses_bounds.npy
    │           └── [images_2|images_4]
    │
    └── co3d               # Link: https://github.com/facebookresearch/co3d
        └── [donut|teddybear|umbrella|...]
            ├── frame_annotations.jgz
            ├── set_lists.json
            └── [129_14950_29917|189_20376_35616|...]
                ├── images
                │   └── frame*.jpg
                └── masks
                    └── frame*.png
</details>

## Instructions

After throughly following the instruction of [DVGO](https://github.com/sunset1995/DirectVoxGO), you can run test the code with below command:

```bash
python run.py --config configs/color_transform_module/rawnerf/scenes/bikes.py --render_test --dump_images
```