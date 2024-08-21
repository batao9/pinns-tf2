# CUDAの公式ベースイメージを使用
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# タイムゾーンの設定を自動化
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# pipのインストール
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# TensorFlowをインストール
RUN pip3.9 install --no-cache-dir tensorflow==2.13.0

# 必要なPythonライブラリをインストール
RUN pip3.9 install --no-cache-dir \
    hydra-core==1.3.2 \
    hydra-colorlog==1.2.0 \
    hydra-optuna-sweeper==1.2.0 \
    rootutils \
    pre-commit \
    rich \
    pytest \
    pyDOE \
    scipy \
    matplotlib

RUN pip3.9 install pinnstf2

# CUDAの環境変数を設定
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

# シンボリックリンクを作成してpython3コマンドをpython3.9に対応させる
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3