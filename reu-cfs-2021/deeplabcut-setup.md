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
