

# REU Computing For Structure:
_A research experience for undergraduate at the University of Miami_

_This is information for the 2021 edition, with a virtual bootcamp from May 17 to May 21._

## Table of Contents

1. [What do we need](#WhatNeed)
2. [Python and Jupyter](#PythonJupyter)
3. [Git for MacOS](#GitMac)
4. [Getting Homebrew for MacOS](#GetHomebrew)
6. [Git for Windows](#GitWindows)
7. [Cygwin for Windows](#Cygwin)
9. [Windows Subsystem for Linux](#WSL)
10. [Git for Linux](#GitLinux)
11. [Ubuntu on a VM](#FullUbuntu)

## <a name="WhatNeed">What do we need?</a>

1. Python
1. Python packages for Scientific computing
1. Jupyter
1. Git and github
1. ssh

We also need unix proficiency,

1. [The Linux Command Line](http://linuxcommand.org/tlcl.php) by William Shotts
2. My [unix mini-course](https://www.cs.miami.edu/home/burt/learning/unixmini/)


### <a name="PythonJupyter">Python and Jupyter</a>

This is the easiest, 

see: https://www.anaconda.com/products/individual

This is the "individual edition", which is the Free, Open Source edition. 

It includes Python 3-point-something. Python version 2 and version 3 are slightly
incompatible. Choose 3 if ever you have a choice.

The anaconda also includes,

1. Python3
1. Jupyter, the notebook/browser interface to the scientific computing world
1. Packages. Lots of packages, such as Numby, Pandas, Matplotlib
1. The conda package manager
1. The conda python virtual environment system

This should run on everything. Mac, Windows and Linux. That is nice.

#### Using Jupyter

Once git has been installed (see below), bring the REU-CFS githup repo to your local machine, 

> `mkdir reu-cfs ; cd to reu-cfs ; git clone https://github.com/burtr/reu-cfs.git`

Then start the notebook,

> `jupyter notebook`

This works on the terminal of MacOS, and from the Anaconda Prompt on windows.

If auto_actitvates as set false:

> `conda config --set auto_activate_False`

then activate it now,

> `conda activate`

If conda can not be found, then use the full pathname, or adjust your path environment variable. 
The activate command is often found at `~/anaconda3/bin/activate`, where the tilde is 
Unix-speak for "my home directory".

For more information, see

> https://docs.anaconda.com/anaconda/install/linux/


## MacOS

### <a name="GitMac">Git: MacOS</a>

The MacOS terminal is basically unix. This is because MacOS was cobbled together
from free and open source unix, with a next generation kernel, then layered with
Apple proprietary code for things they do (or believed they do) better.

__The pieces of MacOS__

1. A Mach micro-kernel (developed at the University of Rochester and CMU)
1. A FreeBSD operating system (a spin-off of Berkeley Systems Devision Unix from 
the University of California at Berkeley)
1. A proprietary desktop built on Quartz and other application packagers
1. IOKit, for the drivers and all, proprietary and based on Objective C.


__Distros__

In the Linux world, diversification is categorized as "distros" (distributions). 
A distro packages up the work of linux developers for distribution. Package managers
allow you to install the packages you want, and keep up-to-date the packages you have.

Apple also has its for of distribution, called the AppStore.

The Homebrew system is a bridge between the Mac and the  distribution system of 
Linux Distros. It is modelled after distro system, but it runs native on Mac.
Homebrew will give access to the entire Linux universe of free and 
open source Software.

_Terminology:_ CLI means Command Line Interface. GUI is Graphical User Interface.

### <a name="GetHomebrew">Getting Homebrew:</a>

see: https://brew.sh/

it says:
``/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"``

but that fails, first you need:
`git -C /usr/local/Homebrew/Library/Taps/homebrew/homebrew-core fetch --unshallow`


### Getting Git with Homebrew


The possibilities for getting Git on the Mac are explained on the git website:
`https://git-scm.com/download/mac`.

> `brew install git ; rehash ; git --version `

__Note:__ The semi-colon on the Unix command line just allows you to 
do on one line what you would do on many lines. The semicolon is a call to action
for the command it precedes. 


## Windows

### <a name="GitWindows">Git: Windows</a>

The possibilities for getting Git on the Mac are explained on the git website:
`https://git-scm.com/download/win`.

A git-bash is installed, which is both the git program, and a unix shell based 
on the bash shell program. 

### <a name="Cygwin">Cygwin</a>

Install Cygwin, www.cygwin.com. You can get git from cygwin and also the important
ssh program.

From the cygwin home page select setup-x86_64.exe. Take any download server, and it will
show a long list of "pending downloads". Just go ahead and install all this.

When done, repeat the visit to cygwin.com and the click on the setup exe, and this time
choose to install git and ssh. There is a selection window. Change the pulldown on the 
upper left for View to Not Installed. Search for git, and it has "skip" in the new column. 
Use the pull-down on the right to change that to a version to install.

Do the same for ssh (called openssh). The proceed to Next/Next and it installs.

Cygwin works as a custom terminal window, and a completely isolated file system branch.
The file system branch it install is a traditional Unix file system. For instance, /usr/local/bin,
and other names familiar to unix users.

When completed, you can a _Cygwin64 Terminal_. The default install leaves an icon for this on the desktop.
Inside this window, you are on what seems to be a unix machine. The window is a unix shell, and the filesystem
is laid out as is familiar to unix programer.

Check the install with the command `which ssh`, and it should return `/usr/bin/ssh`, and `which git` should return `/usr/bin/git`.

### <a name="WSL">Windows Subsystem for Linux</a>

The other possiblity is to turn on Windows long-awaited (it was part of the original design idea of Windows NT) subsystem for linux. 

1. Goto Control Panel --> Programs --> Turn Windows Features On Or Off. 
2. Enable the “Windows Subsystem for Linux” option in the list.
3. Click ok and reboot
4. After reboot install the linux distro using the Microsoft App Store.

When accomplished, you can start and Bash shell, and you will see a standard unix file hierarchy. This is a standard linux install, 
so native Ubuntu packages work. 

The windows files are found in `/mnt/c`, and if there were other drive letters, they to we be found in `/mnt`.

The original Windows NT was intended to run three "flavors" &mdash; win32, os2 and posix. Application level "OS Servers" ran, and
all operating system calls were routed through these servers, and these servers forwarded the service request to the NT Kernel. The original
NT Kernel project as a collaboration. When Microsoft found that it was better off on its own, only the win32 OS server was fully
implemented.



See the [How To Geek](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) for the procedure.


## Linux


### <a name="GitLinux">Git: Linux</a>

On Ubuntu,

> `sudo apt-get install git`


### <a name="FullUbuntu">Full Ubuntu Install</a>

An alternative approach is to create an Ubuntu 18.05 virtual machine.

1. Install Virtual Box for your platform, `https://www.virtualbox.org/`
1. Also install the Extension Pack.
1. Download the Ubuntu 18.04 Desktop ISO, `https://ubuntu.com/download/desktop`
1. **NOTE** I did not have success with Ubuntu 20.
1. Start Virtual Box and select the Ubuntu Desktop ISO.
   1. Under Machine->New
   1. I choose 4096 of memory
   1. I choose 45G of disk (dynamically sized)
   1. When you "start" the created VM you navigate to the ISO file (the icon next to the file choice)
   1. Take all defaults (although maybe adjust location and you need to choose a username).
1. sudo apt-get install build-essential
1. From the VM menu bar, selected Devices->Insert Guest Additions ...
1. Run the CD folder. (It either prompts to run the folder, or there is a button.)
1. Set Devicse->Shared Clipboard as desired (e.g. Bidirectional)
1. Configure Devices->Shared Folders->Shared Folders Settings as desired.
   1. For instance, create a new share from a folder on you host to a folder on your VM.
   1. Auto-mount and Make Permanent 
   1. `sudo adduser $USER vboxsf` to gain permissions on the folder.
1. Reboot

### Install git and get repo

> `sudo apt-get install git ; mkdir git ; cd git ; git clone https://github.com/burtr/reu-cfs`

### Install Anaconda

Download the .sh file from `https://www.anaconda.com/products/individual` and run it,

> `cd Downloads ; chmod a+x Anaconda3-2021.05-Linux-x86_64.sh ; ./Anaconda3-2021.05-Linux-x86_64.sh`

Exit the terminal and reopen the terminal. Navigate to the reu-cfs folder with Jupyter notebooks 
and start Jupyter, 

> `jupyter notebook`





