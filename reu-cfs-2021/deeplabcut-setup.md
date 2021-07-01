# Setting up DeepLabCut on various machines


## On sickles or any bromeliad

<pre>
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
sh Anaconda3-2021.05-Linux-x86_64.sh
git clone https://github.com/DeepLabCut/DeepLabCut.git
cd DeepLabCut/conda-environments/
conda env create -f DLC-GPU-LITE.yaml 
conda activate DLC-GPU-LITE
conda install -c conda-forge ffmpeg-python
cd ../examples
# remove line 318 in testscript.py:   allow_growth=True
rm -rf TEST-Alex-2021-06-08/
python testscript.py
conda deactivate
</pre>


### DLC-GPU-LITE.yaml

The yaml file was removed from the repo on May 28, 2021. Here is the file:

<code>
# DLC-GPU-LITE.yaml

#DeepLabCut2.0 Toolbox (deeplabcut.org)
#© A. & M. Mathis Labs
#https://github.com/DeepLabCut/DeepLabCut
#Please see AUTHORS for contributors.

#https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#Licensed under GNU Lesser General Public License v3.0
#
# DeepLabCut environment
# FIRST: INSTALL CORRECT DRIVER for GPU, see https://stackoverflow.com/questions/30820513/what-is-the-correct-version-of-cuda-for-my-nvidia-driver/30820690
#
# install: conda env create -f DLC-GPU-LITE.yaml
# update:  conda env update -f DLC-GPU-LITE.yaml
name: DLC-GPU-LITE
dependencies:
  - python=3.7
  - pip
  - cudnn=7
  - jupyter
  - nb_conda
  - Shapely
  - pip:
    - tensorflow-gpu==1.15.5
    - deeplabcut
</code>
