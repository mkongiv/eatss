FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

# update packages
RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive TZ=US/Eastern apt-get install tzdata -y

# install essential tools
RUN apt-get install wget -y
RUN apt-get install git -y
RUN apt-get install curl -y

# install tools for isl, pet, and ppcg
RUN apt-get install autoconf -y
RUN apt-get install libtool -y
RUN apt-get install libgmp-dev -y
RUN apt-get install pkg-config -y

# install packages for pet
RUN apt-get install zlib1g-dev -y
RUN apt-get install libncurses-dev -y
RUN apt-get install libxml2-dev -y

# install llvm+clang 8.0
RUN cd /opt/ && \
    wget https://releases.llvm.org/8.0.0/clang+llvm-8.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz && \
    tar -xvf clang+llvm-8.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz && \
    ln -s clang+llvm-8.0.0-x86_64-linux-gnu-ubuntu-18.04 llvm && \
    echo "export PATH=$PATH:/opt/llvm/bin" >> /etc/bash.bashrc 

# set path
ENV PATH "${PATH}:/opt/llvm/bin"

# install clang libraries
RUN cd /opt/llvm && \
    echo "/opt/llvm/lib" > libllvm.conf && \
    mv libllvm.conf /etc/ld.so.conf.d/libllvm.conf

# install isl
RUN cd /opt && \
    git clone git://repo.or.cz/isl.git  && \
    cd isl && \ 
    git checkout isl-0.24 && \
    ./autogen.sh && \
    ./configure && \
    make && \ 
    make install

# install pet
RUN cd /opt && \ 
    git clone git://repo.or.cz/pet.git && \
    cd pet && \
    git checkout pet-0.11.6 && \
    ./autogen.sh && \
    ./configure --with-isl=system && \
    make && \ 
    make install

# install ppcg
RUN cd /opt && \
    git clone git://repo.or.cz/ppcg.git && \
    cd ppcg && \
    git checkout ppcg-0.08.4 && \
    export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/opt/pet && \
    ./autogen.sh && \
    ./configure --with-pet=system --with-isl=system && \
    make && \
    make install && \
    echo "export PATH=$PATH:/opt/ppcg" >> ~/.bashrc 

# install anaconda
RUN cd /tmp/ && \
    curl -O https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh && \
    bash Anaconda3-2023.09-0-Linux-x86_64.sh -b -p /opt/anaconda3 && \
    echo "export PATH=$PATH:/opt/anaconda3/bin" >> ~/.bashrc 

# set path
ENV PATH "${PATH}:/opt/anaconda3/bin"

# copy src
COPY eatss-gpus /home/eatss-gpus

# install anaconda environments
RUN eval "$(/opt/anaconda3/bin/conda shell.bash hook)" && \
    cd /home/eatss-gpus/ && \
    conda create --name eatss --file conda-env-z3-reqs.txt && \
    cd profile && \
    conda create --name eatss-profile --file requirements.txt 

# run the fix for z3-solver
RUN eval "$(/opt/anaconda3/bin/conda shell.bash hook)" && \
    conda activate eatss && \
    pip install z3-solver


