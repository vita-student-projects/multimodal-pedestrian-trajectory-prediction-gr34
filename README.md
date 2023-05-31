## Code Usage:

**Required packages**

For online Python notebook environments the following packages are needed to be installed. Among which the ```nuscenes-devkit``` and ```waymo-open-dataset``` python libraries that provide tools to work with the NuScenes and Waymo datasets. 

``` 
!pip install nuscenes-devkit matplotlib==3.7 waymo-open-dataset-tf-2-11-0==1.5.2

```

We also need to install ```torch-scatter``` package for training which we do like this for it to work on Colab.

```
import torch
!pip uninstall torch-scatter torch-sparse torch-geometric torch-cluster  --y
!pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install git+https://github.com/pyg-team/pytorch_geometric.git
```


Clone the repository ```waymo-motion-prediction-challenge-2022-multipath-plus-plus``` and switch to the specified branch. To be able to import python modules, add the ```code``` directory to the system path. Commands to be run are as follows:

```
!git clone https://github.com/Alvorecer721/waymo-motion-prediction-challenge-2022-multipath-plus-plus multipathpp
!cd multipathpp && git checkout boris/nuscenes-configs

import sys
sys.path.insert(0, '/content/multipathpp/code')
```
**Prerendering**

First we need to prepare data for training. Data preprocessing is done through rendering objects called visualizers created from a configuration file. The ```MultiPathPPRenderer``` class, which is essentially a visualizer, is utilized to perform prerendering tasks. These tasks involve selection of valid agents and road networks, preperation of road network information such as extracting coordinates of nodes, their types and IDs, transformation of data into agent centric coordinate system, filtering closest road elements, generating embeddings for road segments and trajectory classification of agents basewd on motion predictions. 

The prerendering script will convert the original data format into set of ```.npz``` files each containing the data for a single target agent. From ```code``` folder run
```
!python3 multipathpp/code/prerender/prerender_nuscenes.py \
   --data-version v1.0-mini \
   --data-path drive/MyDrive/multipathpp/nuscenes/v1.0-mini \
   --output-path drive/MyDrive/multipathpp/prerendered_nuscenes \
   --config multipathpp/code/configs/nuscenes_prerender.yaml
```
Rendering is a memory consuming process, therefore it uses multiprocessing to speed up the rendering. So, you may want to use ```n-shards > 1``` and running the script a few times using consecutive ```shard-id``` values

**Dataset Splitting**

The prerendered data is split into training and validation based on scene identifiers extracted from the filenames, as such this method assumes a certain ordering and formatting for scene identifiers of the filenames. Run the following commands to do so and to move the datasets to corresponding folders **train** and **val**:

```
import os

PRERENDERED_NUSCENES_PATH = 'drive/MyDrive/multipathpp/prerendered_nuscenes'
train_files = [filename for filename in os.listdir(PRERENDERED_NUSCENES_PATH) if int(filename.split('_')[1]) < 8]
val_files = [filename for filename in os.listdir(PRERENDERED_NUSCENES_PATH) if int(filename.split('_')[1]) >= 8]

os.makedirs(os.path.join(PRERENDERED_NUSCENES_PATH, 'train'))
os.makedirs(os.path.join(PRERENDERED_NUSCENES_PATH, 'val'))

for f in train_files:
 os.rename(os.path.join(PRERENDERED_NUSCENES_PATH, f), os.path.join(PRERENDERED_NUSCENES_PATH, 'train', f))

for f in val_files:
 os.rename(os.path.join(PRERENDERED_NUSCENES_PATH, f), os.path.join(PRERENDERED_NUSCENES_PATH, 'val', f))

!ls drive/MyDrive/multipathpp/prerendered_nuscenes
```

**Normalization**

At this stage the training data is read and certain normalization coefficients are calculated and saved. The normalization coefficients scale the data to a common form, resolving disparicies among different feaures to prevent bias in prediction after the training process. The ```nuscenes_normalization.yaml``` config file specifies the features and their corresponding normalization procedures. The calculated normalization coefficients are saved in a .npy file for use in the training. Run the following commands for normalization:

```
!python3 multipathpp/code/normalization.py \
    --data-path drive/MyDrive/multipathpp/prerendered_nuscenes/train \
    --output-path drive/MyDrive/multipathpp/normalization/normalization_coefs_nuscenes.npy \
    --config multipathpp/code/configs/nuscenes_normalization.yaml
```


**Training**

The ```train.py``` file runs the training process which uses the configuration file ```nuscenes_final_RoP_Cov_Single.yaml```. The validation data ensures the 
To train the model please run the following commands:

```
!python3 multipathpp/code/train.py \
    --train-data-folder drive/MyDrive/multipathpp/prerendered_nuscenes/train \
    --val-data-folder drive/MyDrive/multipathpp/prerendered_nuscenes/val \
    --norm-coeffs drive/MyDrive/multipathpp/normalization/normalization_coefs_nuscenes.npy \
    --config multipathpp/code/configs/nuscenes_final_RoP_Cov_Single.yaml \
    --epoch 40
```

**Scene Data Visualization**

To be able to better grasp the performance of the trained model, it is beneficial to visualize the ground truth and predictions. Following helper functions are used for this purpose where the ```plot_arrowbox``` visualizes an arrow box to represent the vehicle, ```plot_roadlines``` plots the road segments as 2D map and ```plot_scene``` is used to visualize entire scene, combining the previously mentioned functions. 

```
import matplotlib.pyplot as plt

def plot_arrowbox(center, yaw, length, width, color, alpha=1):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array(((c, -s), (s, c))).reshape(2, 2)
    box = np.array([
        [-length / 2, -width / 2],
        [-length / 2, width / 2],
        [length / 2, width / 2],
        [length * 1.3 / 2, 0],
        [length / 2, -width / 2],
        [-length / 2, -width / 2]])
    box = box @ R.T + center
    plt.plot(box[:, 0], box[:, 1], color=color, alpha=alpha)


def plot_roadlines(segments):
    plt.scatter(segments[:, 0, 0], segments[:, 0, 1], color="fuchsia", s=0.5)


def plot_scene(scene_data):
    for timezone, color in [('history', 'blue'), ('future', 'yellow')]:
        for i in range(len(scene_data[f"other/{timezone}/xy"])):
            for other_position, other_yaw, other_valid in zip(
                    scene_data[
                        f"other/{timezone}/xy"][i],
                    scene_data[f"other/{timezone}/yaw"][i],
                    scene_data[f"other/{timezone}/valid"][i]):
                if other_valid.item() == 0:
                    continue
                plot_arrowbox(
                    other_position, other_yaw, scene_data[f"other/{timezone}/length"][i][-1].item(),
                    scene_data[f"other/{timezone}/width"][i][-1].item(), color, alpha=0.5)

    for timezone, color in [('history', 'red'), ('future', 'green')]:
        for target_position, target_yaw, target_valid in zip(
                scene_data[f"target/{timezone}/xy"][0],
                scene_data[f"target/{timezone}/yaw"][0, :, 0],
                scene_data[f"target/{timezone}/valid"][0, :, 0]):
            if target_valid == 0:
                continue
            plot_arrowbox(target_position, target_yaw, scene_data["target/history/length"][0][-1].item(),
                          scene_data["target/history/width"][0][-1].item(), color)

    plot_roadlines(scene_data["road_network_segments"])
```

**Model weights**
- [Waymo](https://drive.google.com/file/d/1zpjXsaHCLJHZvthH_hPw3kEQkYwI81lW/view?usp=sharing) 
- [nuScenes](https://drive.google.com/file/d/1u7teQC_hIoPqcF8BmgQjWSj3bkTc03h_/view?usp=sharing)

**Contributions**
- Yixuan focused on optimising Multipath++ model for pedestrians, trained it on Waymo dataset and integrated Waymo API to calculate the metrics 
- Boris worked on converting nuScenes to Waymo format, trained Multipath++ model on nuScenes and calculated normalization parameters
- Melike specialised in extracting ego vehicle data for nuScenes dataset, helped with training and documented our work
