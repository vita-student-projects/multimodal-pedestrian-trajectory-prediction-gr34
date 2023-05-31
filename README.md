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


First we need to prepare data for training. Data preprocessing is done through rendering objects called visualizers created from a configuration file. The ```MultiPathPPRenderer``` class, which is essentially a visualizer, is utilized to perform prerendering tasks. These tasks involve selection of valid agents and road networks, preperation of road network information such as extracting coordinates of nodes, their types and IDs, transformation of data into agent centric coordinate system, filtering closest road elements, generating embeddings for road segments and trajectory classification of agents basewd on motion predictions. 

The prerendering script will convert the original data format into set of ```.npz``` files each containing the data for a single target agent. To prerender data for nuScenes from ```code``` folder run (assuming your data is stored on google drive)

```
!python3 prerender/prerender_nuscenes.py \
   --data-version v1.0-mini \
   --data-path drive/MyDrive/multipathpp/nuscenes/v1.0-mini \
   --output-path drive/MyDrive/multipathpp/prerendered_nuscenes \
   --config configs/nuscenes_prerender.yaml
```

**Dataset Splitting**

The prerendered data for nuScenes is split into training and validation based on scene identifiers extracted from the filenames, as such this method assumes a certain ordering and formatting for scene identifiers of the filenames. Run the following commands to do so and to move the datasets to corresponding folders **train** and **val**:

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
!python3 normalization.py \
    --data-path drive/MyDrive/multipathpp/prerendered_nuscenes/train \
    --output-path drive/MyDrive/multipathpp/normalization/normalization_coefs_nuscenes.npy \
    --config configs/nuscenes_normalization.yaml
```


**Training**

The ```train.py``` file runs the training process which uses the configuration file ```nuscenes_final_RoP_Cov_Single.yaml```. The validation data ensures the 
To train the model please run the following commands:

```
!python3 train.py \
    --train-data-folder drive/MyDrive/multipathpp/prerendered_nuscenes/train \
    --val-data-folder drive/MyDrive/multipathpp/prerendered_nuscenes/val \
    --norm-coeffs drive/MyDrive/multipathpp/normalization/normalization_coefs_nuscenes.npy \
    --config configs/nuscenes_final_RoP_Cov_Single.yaml \
    --epoch 40
```

**Inference**
To receive the metrics results (and dump them to ```output.txt``` for nuScenes you need to run

```
!python3 inference.py \
    --data-folder drive/MyDrive/multipathpp/prerendered_nuscenes/val \
    --norm-coeffs drive/MyDrive/multipathpp/normalization/normalization_coefs_nuscenes.npy \
    --checkpoint drive/MyDrive/multipathpp/best_single_model.pth \
    --config configs/nuscenes_final_RoP_Cov_Single.yaml
```

**Scene Data Visualization**

To be able to better grasp the performance of the trained model, it is beneficial to visualize the ground truth and predictions. Run this script to get ```trajectories.png``` file with trajectories for nuScenes for the agent number 60

```!python3 pred_visualisation.py \
    --data-folder drive/MyDrive/multipathpp/prerendered_nuscenes/val \
    --norm-coeffs drive/MyDrive/multipathpp/normalization/normalization_coefs_nuscenes.npy \
    --checkpoint drive/MyDrive/multipathpp/best_single_model.pth \
    --config configs/nuscenes_final_RoP_Cov_Single.yaml \
    --agent-idx 60
```

**Model weights**
- [Waymo](https://drive.google.com/file/d/1zpjXsaHCLJHZvthH_hPw3kEQkYwI81lW/view?usp=sharing) 
- [nuScenes](https://drive.google.com/file/d/1u7teQC_hIoPqcF8BmgQjWSj3bkTc03h_/view?usp=sharing)

**Contributions**
- Yixuan focused on optimising Multipath++ model for pedestrians, trained it on Waymo dataset and integrated Waymo API to calculate the metrics 
- Boris worked on converting nuScenes to Waymo format, trained Multipath++ model on nuScenes and calculated normalization parameters
- Melike specialised in extracting ego vehicle data for nuScenes dataset, helped with training and documented our work
