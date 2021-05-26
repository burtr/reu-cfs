## sickles setup

You need to get the download URL. They want you to dl this with your browser, but you can copy link.
I already have the link current at this time. Using that, 

`wget Anaconda3-2021.05-Linux-x86_64.sh`

`bash Anaconda3-2021.05-Linux-x86_64.sh`

and answer the questions in the default, and YES to the running of conda init.

### important

logout and login and verify your path,

`$ which python`

`~/anaconda3/bin/python`

### install gpu and deep learning stuff

It often takes awhile to solve the environment. This might mean you need to adjust the versions of your 
software. It can help to run:

<pre>
conda update conda --all
conda update anaconda
</pre>

I had to run this to install pytorch. They did a few downgrades to get pytorch.

`conda install cudnn`

`conda install tensorflow-gpu`

`conda install pytorch`


