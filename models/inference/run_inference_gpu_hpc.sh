#!/bin/bash
#BSUB -n 4
#BSUB -W 4320
#BSUB -J naip-inf
#BSUB -o stdout.%J
#BSUB -e stderr.%J
#BSUB -q gpu
#BSUB -R span[hosts=1]
#BSUB -R "select[ l40 || h100 || l40s ]"
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -R rusage[mem=64GB]

module load conda
source activate /usr/local/usrapps/rkmeente/talake2/naip_ailanthus_env
export PYTHONPATH=$PYTHONPATH:/home/talake2
export CUDA_LAUNCH_BLOCKING=1
python tiled_inference_uncertainty_gpu_opt_hpc.py
conda deactivate