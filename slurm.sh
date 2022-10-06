#!/bin/bash -e

[ -d "$HOME"/.ssh ] || mkdir "$HOME"/.ssh
[ -d "$HOME"/.pyterrier ] || mkdir "$HOME"/.pyterrier
[ -d "$HOME"/.ir_datasets ] || mkdir "$HOME"/.ir_datasets
srun \
  --cpus-per-task 4 \
  --mem=50G \
  --gres=gpu:ampere:1 \
  --container-writable \
  --container-image=nvidia/cuda:11.3.1-base-ubuntu20.04 \
  --container-name=fare-ubuntu20.04-cuda11.3.1-python3.9-jdk11 \
  --container-mounts="$PWD":/workspace,"$HOME"/.ssh:/root/.ssh,"$HOME"/.pyterrier:/root/.pyterrier,"$HOME"/.ir_datasets:/root/.ir_datasets \
  --chdir "$PWD" \
  --pty \
  su -c "cd /workspace &&
    apt-get update -y &&
    apt-get install -y openjdk-11-jdk software-properties-common git &&
    add-apt-repository -y ppa:deadsnakes/ppa &&
    apt-get install -y python3.9 python3-pip python3.9-dev &&
    export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/ &&
    pip install cython pipenv &&
    pipenv install --deploy &&
    pipenv run python -m fare $1"
