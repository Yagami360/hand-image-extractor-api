#!/bin/sh
set -eu

if [ ! -e "checkpoints/universal_trained.pth" ] ; then
    cd checkpoints
    FILE_ID=1sWJ54lCBFnzCNz5RTCGQmkVovkY9x8_D
    FILE_NAME=universal_trained.pth
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
    CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}
fi