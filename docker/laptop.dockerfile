FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Please contact with me if you have problems
LABEL maintainer="Zipeng Dai <daizipeng@bit.edu.cn>"
# TODO：网络不好的话可以走代理
ENV http_proxy=http://127.0.0.1:8889
ENV https_proxy=http://127.0.0.1:8889

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash
ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    locales git tmux gedit vim openmpi-bin openmpi-common libopenmpi-dev libgl1 libglx-mesa0 libsm6 libice6 \
    libcanberra-gtk-module libcanberra-gtk3-module libusb-1.0-0 libusb-1.0-0-dev libglib2.0-0  libxext6 \
    libxrender-dev
RUN apt-get install -y --no-install-recommends python3-dev python3-wheel python3-pip

RUN pip3 install --upgrade pip \
    && pip3 install   \
        gradio        \
        opencv-python \
        supervision   \
        mmengine      \
        setuptools    \
        openmim       
RUN  pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
        wheel         \
        torch         \
        torchvision   \
        torchaudio
RUN mim install mmcv==2.0.0

WORKDIR /workspace
RUN mkdir dosod_weights/ && cd dosod_weights && \
    wget https://huggingface.co/D-Robotics/DOSOD/resolve/main/dosod_mlp3x_l.pth && \
    wget https://huggingface.co/D-Robotics/DOSOD/resolve/main/dosod_mlp3x_m.pth && \
    wget https://huggingface.co/D-Robotics/DOSOD/resolve/main/dosod_mlp3x_s.pth
    
RUN rm -rf /var/lib/apt/lists/* && apt-get clean
CMD ["/bin/bash"]