import os


QSUB_SCIPRT = f"""#!/bin/bash 
#PBS -N Enformer_Binary_Prediction_fine_tune_|cell_type|
#PBS -P zk16
#PBS -q gpuvolta 
#PBS -l jobfs=128GB
#PBS -l walltime=24:00:00
#PBS -l mem=128GB
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l wd
#PBS -l storage=scratch/zk16+gdata/zk16
#PBS -M z.li@victorchang.edu.au
#PBS -m abe
#PBS -o /g/data/zk16/zelun/z_li_hon/wonglab_github/enformer-pytorch-fine-tune/finetune/logs
#PBS -e /g/data/zk16/zelun/z_li_hon/wonglab_github/enformer-pytorch-fine-tune/finetune/logs

source /g/data/zk16/zelun/miniconda3/bin/activate enformer-fine-tune
export PYTHONPATH="/g/data/zk16/zelun/z_li_hon/wonglab_github/enformer-pytorch-fine-tune:$PYTHONPATH"
cd /g/data/zk16/zelun/z_li_hon/wonglab_github/enformer-pytorch-fine-tune
python3 finetune/fine_tune_tidy_tanh.py --cell_type='|cell_type|' > "/g/data/zk16/zelun/z_li_hon/wonglab_github/enformer-pytorch-fine-tune/finetune/best_result/|cell_type|/final_|cell_type|_test_result.txt"
"""


if __name__ == "__main__":
    # cell_types = ["allantois", "endothelium", "exe_endo", "gut", "mid_hindbrain", "neuralcrest", "paraxial_meso", "somitic_meso", "surface_ecto", "cardiom", "erythroid", "forebrain", "mesenchyme", "NMP", "pharyngeal_meso", "spinalcord"]
    # cell_types = ["mixed_meso"]
    # cell_types = ['allantois', 'endothelium', 'gut', 'neuralcrest'] #exe_endo
    # cell_types = ['pharyngeal_meso']
    cell_types = ['mixed_meso']
    # cell_types = ['gut']
    for cell_type in cell_types:
        with open(f"qsub_scripts/{cell_type}.sh", "w") as f:
            f.write(QSUB_SCIPRT.replace("|cell_type|", cell_type))
        # run the qsub command
        os.system(f"qsub qsub_scripts/{cell_type}.sh")
