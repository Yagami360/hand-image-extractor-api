#!/bin/sh
set -eu
if [ ! -e "checkpoints/universal_trained.pth" ] ; then
    sh download_model.sh
fi

python app.py \
    --host 0.0.0.0 --port 5001 \
    --device gpu \
    --use_amp \
    --debug
