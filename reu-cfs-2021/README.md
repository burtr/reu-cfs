

# REU Computing For Structure:
_A research experience for undergraduate at the University of Miami_


For 2021 edition. 

What do we need?
1. Python
2. Python packages for Scientific computing
3. Jupyter
4. Git and github

How to get it?

Depends on your platform. Are you Linux, MacOS or Windows?

MACOS Git

see: https://git-scm.com/download/mac

There are choices here. XCode is great, but it is a long download form the AppStore 
and is picky about your MACOS version. Even after install you will have to do some
"install CLI tools" magic. CLI means Command Line Interface

Homebrew ports lots of Linux stuff to MacOS.

Do you know how to get a terminal window on Mac? Did you know that Mac's are
a blend of Linux? Well of Unix really, and Linux is a Unix. But everyone just says
Linux. But Mac OS is four things:

1. A Mach micro-kernel (developed at the University of Rochester and CMU)
2. A FreeBSD operating system (a spin-off of Berkeley Systems Devision Unix from 
the University of California at Berkeley)
3. A proprietary desktop built on Quartz and other application packagers
4. IOKit, for the drivers and all, proprietary and based on Objective C.

Getting Homebrew:

see: https://brew.sh/

it says:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

but that fails, first you need:
git -C /usr/local/Homebrew/Library/Taps/homebrew/homebrew-core fetch --unshallow

then you get:
brew install git



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


