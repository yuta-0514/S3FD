#!/bin/bash

CONTAINER_NAME=s3fd
IMAGES=yuta0514/s3fd
TAGS=1.8
PORT=8888

docker run --rm -it --gpus all --ipc host -v ~/dataset:/mnt -v $PWD:/S3FD_pytorch -v ~/.dockerssh:/root/.ssh:ro -p ${PORT}:${PORT} --name ${CONTAINER_NAME} ${IMAGES}:${TAGS}
