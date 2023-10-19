## EventHPE: Event-based 3D Human Pose and Shape Estimation

Shihao Zou, Chuan Guo, Xinxin Zuo, Sen Wang, Xiaoqin Hu, Shoushun Chen, Minglun Gong and Li Cheng. ICCV 2021.

### Dataset
You can download the data from [Google Drive](https://drive.google.com/drive/folders/11gMj-5sgSiBciWNR0V6r9PMpru84zMk5?usp=sharing) 
or [Microsoft OneDrive](https://ualbertaca-my.sharepoint.com/:u:/g/personal/szou2_ualberta_ca/EWZFehf_UdFMiA0TJPdfaiwBSoChTOkZeckoBM8EqbLUOg?e=RkoOL3), 
which consists of
- preprocessed data
  - events_256 (event frames converted from raw events data, resolution 256x256)
  - full_pic_256 (gray-scale images)
  - pose_events (annotated poses of gray-scale images)
  - hmr_results (inferred poses of gray-scale images using [HMR](https://github.com/akanazawa/hmr))
  - vibe_results_0802 (inferred poses of gray-scale images using [VIBE](https://github.com/mkocabas/VIBE))
  - pred_flow_events_256 (inferred optical flow from event frames)
  - model (train/test on a snippet of 8 frames)
- raw events data (Please contact Shihao Zou szou2@ualberta.ca for the access.)

> tar -xf your_file.tar # how to uncompress the provided data

### Requirements
```
python 3.7.5
torch 1.7.0
opendr 0.78 (for render SMPL shape, installed successfully only under ubuntu 18.04)
cv2 4.1.1
```

To download the SMPL model go to [this](https://smpl.is.tue.mpg.de/) project website and 
register to get access to the downloads section. Place under __./smpl_model__. The model 
version used in our project is
```
basicModel_m_lbs_10_207_0_v1.0.0.pkl
basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```

#### An Example environment setup

```bash
docker run --ipc=host --gpus all -dt --name eventHPE -v /home/rowan/dataset:/root/dataset wgeng/ubuntu18.04-cuda110-conda
# enter the container

# tips when setup the environment
conda init bash
conda create --name eventHPE python=3.7.5
conda activate eventHPE
pip install plyfile joblib
pip install torch==1.7.0 torchvision==0.8.1 opencv-python==4.1.1.26 numpy Cython
# pip install torch==1.7.0+cu110 torchvision==0.8.0+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

apt-get install -y build-essential libglu1-mesa-dev libgl1-mesa-dev libcairo2-dev libglfw3-dev libgtest-dev libosmesa6-dev # run this before install opendr
pip install opendr
pip install opendr-toolkit
```