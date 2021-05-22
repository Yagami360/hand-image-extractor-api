# Use Caffe2 image as parent image
#FROM caffe2/caffe2:snapshot-py2-cuda9.0-cudnn7-ubuntu16.04
FROM caffe2/caffe2:snapshot-py2-cuda8.0-cudnn7-ubuntu16.04

RUN mv /usr/local/caffe2 /usr/local/caffe2_build
ENV Caffe2_DIR /usr/local/caffe2_build

ENV PYTHONPATH /usr/local/caffe2_build:${PYTHONPATH}
ENV LD_LIBRARY_PATH /usr/local/caffe2_build/lib:${LD_LIBRARY_PATH}

# Clone the Detectron repository
RUN git clone https://github.com/facebookresearch/densepose /densepose

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install scikit-build
RUN pip install opencv-python==4.2.0.32
RUN pip install -r /densepose/requirements.txt
RUN pip install PyYAML==3.12

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN pip install cython
RUN pip install matplotlib==2.1.0
#RUN pip install python-tk
RUN make install

# for api
RUN pip install tqdm
RUN pip install pillow==6.2.1
RUN pip install flask
RUN pip install flask-cors
RUN pip install requests

# Go to Densepose root
WORKDIR /densepose

# Set up Python modules
RUN make

# [Optional] Build custom ops
RUN make ops

# Fetch DensePose data.
RUN apt update && apt install wget
RUN bash /densepose/DensePoseData/get_densepose_uv.sh
RUN mv UV_data /densepose/DensePoseData/

# For training, download the DensePose-COCO dataset:
RUN bash /densepose/DensePoseData/get_DensePose_COCO.sh
RUN mv DensePose_COCO /densepose/DensePoseData/

# For evaluation, get the necessary files:
RUN bash /densepose/DensePoseData/get_eval_data.sh
RUN mv eval_data /densepose/DensePoseData/
