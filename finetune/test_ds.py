import sys
from data_sets.mouse_8_25 import mouse_8_25

# TODO: note to load the LD_LIB_PATH -> which is on this site
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/g/data/zk16/zelun/miniconda3/envs/enformer-eval/lib
# ref https://stackoverflow.com/questions/58424974/anaconda-importerror-usr-lib64-libstdc-so-6-version-glibcxx-3-4-21-not-fo

# export PYTHONPATH="/g/data/zk16/zelun/z_li_hon/wonglab_github/enformer-pytorch-fine-tune:$PYTHONPATH"
# python3 your_script.py


face_dataset = mouse_8_25()
# fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample)

    if i == 3:
        break
