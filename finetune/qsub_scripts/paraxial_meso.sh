#!/bin/bash 
#PBS -N Enformer_Binary_Prediction_fine_tune_paraxial_meso
#PBS -P zk16
#PBS -q gpuvolta 
#PBS -l jobfs=128GB
#PBS -l walltime=24:00:00
#PBS -l mem=128GB
#PBS -l ngpus=2
#PBS -l ncpus=24
#PBS -l wd
#PBS -l storage=scratch/zk16+gdata/zk16
#PBS -M z.li@victorchang.edu.au
#PBS -m abe
#PBS -o /g/data/zk16/zelun/z_li_hon/wonglab_github/enformer-pytorch-fine-tune/finetune/logs
#PBS -e /g/data/zk16/zelun/z_li_hon/wonglab_github/enformer-pytorch-fine-tune/finetune/logs

source /g/data/zk16/zelun/miniconda3/bin/activate enformer-fine-tune
export PYTHONPATH="/g/data/zk16/zelun/z_li_hon/wonglab_github/enformer-pytorch-fine-tune:$PYTHONPATH"
cd /g/data/zk16/zelun/z_li_hon/wonglab_github/enformer-pytorch-fine-tune
python3 finetune/fine_tune_tidy.py --cell_type='paraxial_meso' > "/g/data/zk16/zelun/z_li_hon/wonglab_github/enformer-pytorch-fine-tune/finetune/best_result/paraxial_meso/final_paraxial_meso_test_result.txt"