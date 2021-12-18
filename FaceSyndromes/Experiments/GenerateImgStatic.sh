#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=8
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:p100:2 --mem=24g --cpus-per-task=24 

#----------------------------------------------------------------------------
# ! generate images, using labels indexing
# ! let's try same random vector, but different label class

cd /data/duongdb/stylegan3-FaceSyndromes

truncationpsi=1 # @trunc=0.7 is recommended on their face dataset. 

for group in '4' '5' '6' '7' '8'
do 

  for class1 in '0' '3' # '5' '6' '7' '8'
  do 

    class=$class1','$group',9'

    for modelname in TfStyleGAN3LabelRes256Up10+Gender/00004-stylegan2-TfStyleGAN3LabelRes256Up10+Gender-gpus2-batch32-gamma1-multilabel/network-snapshot-000760.pkl
    do

      outdir=/data/duongdb/WS22qOther_12082021/Stylegan3Model/$modelname'Interpolate'
      mkdir $outdir

      model=/data/duongdb/WS22qOther_12082021/Stylegan3Model/$modelname

      python3 gen_images.py --outdir=$outdir/$class'T'$truncationpsi'Static' --trunc=$truncationpsi --seeds=0-300 --class=$class --network $model 

    done 
  done 
done 

