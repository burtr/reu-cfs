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

It often takes awhile to solve the environment. Not more than five minutes, but it can seem very long.

`conda install cudnn`

`conda install tensorflow-gpu`

`conda install pytorch`


