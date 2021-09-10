#!/bin/bash
##SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --account=owner-guest
#SBATCH --partition=q04
##SBATCH --gres=gpu:3
#SBATCH --nodelist=g41
#SBATCH --gres=gpu:1
##SBATCH --mem-per-gpu=30G
##SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH -o /share/home/zhongzisha/cluster_logs/coltran-job-train-%j-%N.out
#SBATCH -e /share/home/zhongzisha/cluster_logs/coltran-job-train-%j-%N.err

echo "job start `date`"
echo "job run at ${HOSTNAME}"
nvidia-smi

df -h
nvidia-smi
ls /usr/local
which nvcc
which gcc
which g++
nvcc --version
gcc --version
g++ --version

env

nvidia-smi

free -g
top -b -n 1

uname -a

sleep 1200000000000000000

source venv_test/bin/activate

#export LD_LIBRARY_PATH=$HOME/gcc-7.5.0/install/lib64:$LD_LIBRARY_PATH
#export PATH=$HOME/gcc-7.5.0/install/bin:$PATH
export LD_LIBRARY_PATH=$HOME/glibc2.30/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
# export CUDA_ROOT=$HOME/cuda-10.2-cudnn-7.6.5
# export LD_LIBRARY_PATH=$CUDA_ROOT/libs/lib64:$CUDA_ROOT/lib64:$CUDA_ROOT/lib64/stubs:$LD_LIBRARY_PATH
export CUDA_ROOT=$HOME/cuda-10.2-cudnn-8.2.2
export CUDA_PATH=$CUDA_ROOT
export LD_LIBRARY_PATH=$CUDA_ROOT/libs/lib64:$CUDA_ROOT/lib64:$CUDA_ROOT/lib64/stubs:$LD_LIBRARY_PATH
export PATH=$CUDA_ROOT/bin:$PATH
export CUDA_INSTALL_DIR=$HOME/cuda-10.2-cudnn-8.2.2
export CUDNN_INSTALL_DIR=$HOME/cuda-10.2-cudnn-8.2.2
export TRT_LIB_DIR=$HOME/cuda-10.2-cudnn-8.2.2/TensorRT-8.0.1.6/lib


if [ ${HOSTNAME} == "g39" ]; then

echo ${HOSTNAME}

LOGDIR=`pwd`/logs
DATA_DIR=`pwd`/images/
DATA_DIR_TEST=`pwd`/images_test/
# DATA_DIR_TEST=`pwd`/images_test_gray/

DO_TRAIN_STEP1=0
DO_TRAIN_STEP2=0
DO_TRAIN_STEP3=0
DO_VAL_STEP1=0
DO_VAL_STEP2=0
DO_VAL_STEP3=0
DO_SAMPLE_STEP1=1
DO_SAMPLE_STEP2=1
DO_SAMPLE_STEP3=1

#####################  step 1 ##############################
if [ $DO_TRAIN_STEP1 -ge 1 ]; then
  rm -rf $LOGDIR/colorizer_ckpt_dir
  CUDA_VISIBLE_DEVICES=0 python -m coltran.run \
  --config=coltran/configs/colorizer.py \
  --mode=train \
  --logdir=$LOGDIR/colorizer_ckpt_dir \
  --dataset=custom \
  --data_dir=$DATA_DIR
fi

if [ $DO_VAL_STEP1 -ge 1 ]; then
  CUDA_VISIBLE_DEVICES=0 python -m coltran.run \
  --config=coltran/configs/colorizer.py \
  --mode=eval_valid \
  --logdir=$LOGDIR/colorizer_ckpt_dir \
  --dataset=custom \
  --data_dir=$DATA_DIR
fi

if [ $DO_SAMPLE_STEP1 -ge 1 ]; then
  CUDA_VISIBLE_DEVICES=0 python -m coltran.sample \
  --config=coltran/configs/colorizer.py \
  --mode=sample_test \
  --logdir=$LOGDIR/colorizer_ckpt_dir \
  --dataset=custom \
  --data_dir=$DATA_DIR_TEST

  python coltran/convert_tfrecords_to_png.py $LOGDIR/colorizer_ckpt_dir/gen_data_dir/ 64
fi

#####################  step 2 ##############################
if [ $DO_TRAIN_STEP2 -ge 1 ]; then
  rm -rf $LOGDIR/color_upsampler_ckpt_dir
  CUDA_VISIBLE_DEVICES=0 python -m coltran.run \
  --config=coltran/configs/color_upsampler.py \
  --mode=train \
  --logdir=$LOGDIR/color_upsampler_ckpt_dir \
  --dataset=custom \
  --data_dir=$DATA_DIR
fi


if [ $DO_SAMPLE_STEP2 -ge 1 ]; then
  CUDA_VISIBLE_DEVICES=0 python -m coltran.sample \
  --config=coltran/configs/color_upsampler.py \
  --mode=sample_test \
  --logdir=$LOGDIR/color_upsampler_ckpt_dir \
  --dataset=custom \
  --data_dir=$DATA_DIR_TEST

  python coltran/convert_tfrecords_to_png.py $LOGDIR/color_upsampler_ckpt_dir/gen_data_dir/ 64
fi


#####################  step 3 ##############################
if [ $DO_TRAIN_STEP3 -ge 1 ]; then
  rm -rf $LOGDIR/spatial_upsampler_ckpt_dir
  CUDA_VISIBLE_DEVICES=1 python -m coltran.run \
  --config=coltran/configs/spatial_upsampler.py \
  --mode=train \
  --logdir=$LOGDIR/spatial_upsampler_ckpt_dir \
  --dataset=custom \
  --data_dir=$DATA_DIR

fi

if [ $DO_SAMPLE_STEP3 -ge 1 ]; then
  CUDA_VISIBLE_DEVICES=0 python -m coltran.sample \
  --config=coltran/configs/spatial_upsampler.py \
  --mode=sample_test \
  --logdir=$LOGDIR/spatial_upsampler_ckpt_dir \
  --dataset=custom \
  --data_dir=$DATA_DIR_TEST

  python coltran/convert_tfrecords_to_png.py $LOGDIR/spatial_upsampler_ckpt_dir/gen_data_dir/ 256
fi


fi



