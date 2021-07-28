FROM python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r requirements.txt
COPY . . 
WORKDIR /app/seg/segTMP_pytorch
CMD ls && python3 main_SegTMP.py