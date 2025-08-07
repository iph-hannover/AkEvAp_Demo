# AkEvAp Demo

## Environment
- Create conda environment
  - ```conda create -n akevap```
  - ```conda activate akevap```
- Install Packages
  - ```conda install python=3.12 pillow requests flask python-mss pandas```
  - ```conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia```
  - ```pip install PyQt5 imutils imageio[ffmpeg]```
  - For using PyQt5 Designer install the [latest wheel for pyqt5-plugins](https://pypi.org/project/pyqt5-plugins/#files), which currently only supports Python 3.11. Best is to install it in a separate environment using python 3.11. After running ```pip install pyqt5-tools``` the designer can be started using ```qt5-tools designer```.
  - ```pip install einops timm==0.6.7 smplx chumpy scikit-image yacs``` (see [tram/install.sh](tram/install.sh), but we do not need every dependency from there)
  - ```pip install git+https://github.com/mattloper/chumpy.git@refs/pull/52/head``` (fix [broken chumpy for python>=3.11](https://github.com/mattloper/chumpy/issues/51)
  - ```pip install -r hand_object_detector/requirements.txt```
  - optional: ```conda install 'numpy<1.24'``` (in case ```AttributeError: module 'numpy' has no attribute 'str'.``` is observed)
- Compile the cuda dependencies for hand-object contact estimation:
  - ```conda install gxx=12 "setuptools<80"```
  - ```cd hand_object_detector/lib```
  - ```python setup.py build develop```
- Download public model for hand-object contact estimation:
  - Place [model](https://drive.google.com/open?id=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE) in [hand_object_detector/models/res101_handobj_100K/pascal_voc/](hand_object_detector/models/res101_handobj_100K/pascal_voc/)
- Download relevant models for human pose estimation: (see [tram/README.md](tram/README.md#prepare-data))
  - Place models from [SMPLify](https://smplify.is.tue.mpg.de) and [SMPL](https://smpl.is.tue.mpg.de) into [tram/data/smpl](tram/data/smpl)
  - Place [model](https://drive.google.com/file/d/1fdeUxn_hK4ERGFwuksFpV_-_PHZJuoiW/view?usp=share_link) into [tram/data/pretrain](tram/data/pretrain)

## Streaming Server
Install [srt-live-server](https://github.com/Edward-Wu/srt-live-server) in _Windows Subsystem for Linux (WSL)_
- Install a distribution in WSL, e.g. ```wsl --install -d openSUSE-Tumbleweed```
- Install required packages ```sudo zypper install git make cmake gcc-c++ openssl-devel tcl```

- Clone repositories
  - ```sudo git clone https://github.com/Edward-Wu/srt-live-server```
  - ```sudo git clone https://github.com/Haivision/srt```
- Compile _libsrt_
  ```
  cd srt
  #cmake ./
  ./configure  # easiest way when using openssl (requires tcl)
  make -j 16
  sudo make install
  sudo ldconfig  # update linker cache
  cd ..
  ```
- Compile _srt-live-server_ (some fixes are required as make will fail otherwise)
  - in Makefile change: ```INC_FLAGS``` -> ```INC_PATH```
  - in slscore/common.cpp add: ```#include <ctime>```
  ```
  cd srt-live-server
  make -j 16
  ```

- Configure and run _srt-live-server_
  - in sls.conf:
    ```
            domain_player myserver;
            domain_publisher myserver;
                app_player stream ;           
                app_publisher live ;
    ```
  - ```./bin/sls -c sls.conf```

- Start and play a stream using [ffmpeg](https://ffmpeg.org/) (a version compiled with srt support is required, e.g. https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z)
  ```
  ./ffmpeg.exe -re -i .\avideo.mp4 -c copy -f mpegts "srt://localhost:8080?streamid=myserver/live/test"
  ./ffplay.exe -fflags nobuffer -i "srt://localhost:8080?streamid=myserver/stream/test"
  ```
