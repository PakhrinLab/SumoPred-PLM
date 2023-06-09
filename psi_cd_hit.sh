#!/bin/bash
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
module load BLAST+/2.9.0-gompi-2019b
module load Perl/5.30.0-GCCcore-8.3.0
./cdhit-4.8.1/psi-cd-hit/psi-cd-hit.pl -i test_fullsequence.fasta -o SUBASH_CHMOD_CD_HIT_30_Percent_test_general_with_CD_HIT.fasta -c 0.3
