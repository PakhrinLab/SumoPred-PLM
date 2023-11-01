#!/bin/bash
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=wsu_gen_gpu.q
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
source ~/virtualenv/serena_sleeping/bin/activate


python ~/Independent_Test_ANN_Sumoylation_500_Times.py