

# REU Computing For Structure:
_A research experience for undergraduate at the University of Miami_

_This is information for the 2021 edition, with a virtual bootcamp from May 17 to May 21._

## What do we need?

1. Python
1. Python packages for Scientific computing
1. Jupyter
1. Git and github


### Python and Jupyter

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




### Git


#### MacOS

MacOS is a form of Linux, and that is totally badly said. In fact, MacOS is,

1. A Mach micro-kernel (developed at the University of Rochester and CMU)
1. A FreeBSD operating system (a spin-off of Berkeley Systems Devision Unix from 
the University of California at Berkeley)
1. A proprietary desktop built on Quartz and other application packagers
1. IOKit, for the drivers and all, proprietary and based on Objective C.

The terminal window in MacOS gives access to the FreeBSD operating system, and is 
therefore a Unix shell. More about Unix in my short tutorial.

see: https://www.cs.miami.edu/home/burt/learning/unixmini/

In that tutorial, you learn that Linux is a kernel and operating system based on 
Unix, as is MacOS, so there is a lot of shared concepts. In the Linux world, 
diversification is categorized as "distros" (distributions). A distro packages up
the work of linux/unix developers for distribution. An amazing infrastructure of 
repositories and package managers has evolved.

Apple also has its for of distribution, called the AppStore.

The Homebrew system is a bridge between the Apple distribution system and the 
distribution system of Linux Distros. 

The possibilities for getting Git are explained on the git website:
 https://git-scm.com/download/mac

We will try to get it through the Homebrew system. If mastered, Homebrew will 
give access to the entire Linux universe of Free, Open Source Software.


_Terminology:_ CLI means Command Line Interface. GUI is Graphical User Interface.

#### Getting Homebrew:

see: https://brew.sh/

it says:
``/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"``

but that fails, first you need:
`git -C /usr/local/Homebrew/Library/Taps/homebrew/homebrew-core fetch --unshallow`


#### Getting Git

   brew install git
   rehash
   git --version 


## Other stuff



mkdir git and cd to git
then clone the repo:

> git clone https://github.com/burtr/reu-cfs.git

Jupyter should be installed in your lab machines and will start with:

> jupyter notebook

from the command line.

If auto_actitvates as set false:

> conda config --set auto_activate_False

then activate it now,

> conda activate

If conda can not be find, then use the full path

> /usr/local/anaconda3/bin/activate
conda init # is this needed?

For more information, see

> https://docs.anaconda.com/anaconda/install/linux/


