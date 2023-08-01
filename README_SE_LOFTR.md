# SELoFTR: How to train


### 1: Dataset Preparation

Download MegaDepth dataset
```commandline
wget https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz
wget https://www.cs.cornell.edu/projects/megadepth/dataset/MegaDepth_SfM/MegaDepth_SfM_v1.tar.xz
```

Download MegaDepth dataset indices from [google drive](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf)
```commandline
unzip train-data-20230727T160329Z-003.zip
unzip testdata-20230727T160337Z-001.zip
tar -xvf train-data/megadepth_indices.tar
tar -xvf testdata/megadepth_test_1500.tar
```

Clone D2-Net repo and make sure to install the dependencies
```commandline
conda activate loftr
git clone https://github.com/mihaidusmanu/d2-net.git
cd d2-net/megadepth_utils

conda install h5py imageio imagesize matplotlib scipy tqdm
conda install -c conda-forge colmap
# Downgrade numpy to avoid errors saving numpy files
pip install numpy==1.21.6
```

Undistort images with D2-Net
```commandline
python undistort_reconstructions.py --colmap_path '' --base_path /path/to/megadepth
bash preprocess_undistorted_megadepth.sh /path/to/megadepth /path/to/output/folder
```

Build symlinks
```commandline
cd /path/to/LoFTR/
ln -s /path/to/megadepth/phoenix data/megadepth/train/phoenix
ln -s /path/to/megadepth/phoenix data/megadepth/test/phoenix
ln -s /path/to/megadepth_d2net/Undistorted_SfM data/megadepth/train/Undistorted_SfM
ln -s /path/to/megadepth_d2net/Undistorted_SfM data/megadepth/test/Undistorted_SfM
ln -s /path/to/megadepth_indices/* data/megadepth/index
```


### 2: Training
```commandline
scripts/train_se_loftr/outdoor_ds.sh
```

### KNOWN ISSUES

* any pytorch version works fine but ensure that cuda works (`torch.cuda.is_available()`)
* make sure to install pytorch_lightning 1.3.5 version
* if scipy complains about an ImportError regarding libstdc++.so.6 or other library loading issues
```commandline
conda deactivate
conda remove -n loftr --all
conda create -n loftr python=3.8
conda activate loftr
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
