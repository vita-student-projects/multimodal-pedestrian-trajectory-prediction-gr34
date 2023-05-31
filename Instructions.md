---
title: Instructions on configuring Conda environment in SCITAS
author: Yixuan Xu
date: 2023-05-18

---

Firstly, acquire Miniconda by downloading it (Avoid attempting a direct pip install since we will later need to download PyTorch, and using pip may result in the command being killed). ***Restart*** the terminal after running the following code:

```bash
curl -sL \
  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
  "Miniconda3.sh"
  
bash Miniconda3.sh
```

Using miniconda to create virtual environment, please verify the loaded python version: 3.8.16

```bash
conda env create -f environment.yaml
conda activate dlav
```

(Optional) Using SCITAS with Jupyter Notebook

```bash
# Create Jupyter Kernel
python -m ipykernel install --user --name dlav --display-name "civil459"

# Running Jupyter notebook on your own Browser
jupyter notebook --ip="$(hostname -s).epfl.ch"
```

