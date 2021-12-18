



script = """#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0


datapath=/data/duongdb/WS22qOther_12082021

img_folder=$datapath/IMG_FOLDER

outdir=$datapath/Stylegan3Model/OUT_FOLDER

# ! note, loading small models: https://github.com/NVlabs/stylegan3/issues/23#issuecomment-970706600
# ! https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3/files
# ! load stylegan3-r-ffhqu-256x256.pkl or stylegan3-t-ffhqu-256x256.pkl? we want ffhq-aligned though?? not the ffhqu-unaligned
# ! load stylegan3-t-ffhq-1024x1024.pkl too ?

nvidia_pretrain=/data/duongdb/stylegan3-FaceSyndromes/NvidiaPretrainedModel/stylegan3-t-ffhqu-256x256.pkl

# ! default in paper FFHQ-U ablations at 256x256 resolution, --cfg=stylegan3-t --gpus=8 --batch=64 --gamma=1 --mirror=1 --aug=noaug --cbase=16384 --dlr=0.0025

# ! metface at 1024 --cfg=stylegan3-t --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=10 --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl

# ! batch=32 at res=256 may be okay with 2gpus, batch 24 takes 10/16gb 
# ! batch=16 at res=1024 hits 15/16gb with 2 gpus 

# ! train 
cd /data/duongdb/stylegan3-FaceSyndromes

python train.py --outdir=$outdir --data=$img_folder \
--resume=$nvidia_pretrain \
--cfg=stylegan3-t \
--gpus=2 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \
--aug=ada \
--cond=True


"""


import time,os,sys,re,pickle
from datetime import datetime

now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y%H%M%S")

os.chdir('/data/duongdb/WS22qOther_12082021')
counter = 1

IMG_FOLDER = 'TfStyleGAN3LabelRes256'
OUT_FOLDER = IMG_FOLDER

newscript = re.sub('IMG_FOLDER',IMG_FOLDER,script)
newscript = re.sub('OUT_FOLDER',OUT_FOLDER,newscript)
fname = 'script'+str(counter+1)+date_time+'.sh'
fout = open(fname,'w')
fout.write(newscript)
fout.close()
counter = counter + 1
# time.sleep(1)
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:p100:2 --mem=24g --cpus-per-task=24 
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:v100x:2 --mem=24g --cpus-per-task=24 
os.system('sbatch --partition=gpu --time=48:00:00 --gres=gpu:p100:2 --mem=16g --cpus-per-task=12 '+fname)


