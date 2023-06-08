FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

RUN sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list
RUN sed -i s/ports.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list

# To use a different model, change the model URL below:
ARG MODEL_URL="https://firebasestorage.googleapis.com/v0/b/new-test-project-b425d.appspot.com/o/asserts%2Fmeinamix_meinaV10.safetensors?alt=media"
ARG CONTROL_URL="https://firebasestorage.googleapis.com/v0/b/new-test-project-b425d.appspot.com/o/asserts%2Fcontrol_v11p_sd15_lineart.pth?alt=media"

ENV MODEL_URL=${MODEL_URL}
ENV CONTROL_URL=${CONTROL_URL}
ENV HF_TOKEN=${HF_TOKEN}

RUN apt update && apt-get -y install git wget \
    python3.10 python3.10-venv python3-pip \
    build-essential libgl-dev libglib2.0-0 wget
RUN ln -s /usr/bin/python3.10 /usr/bin/python

RUN useradd -ms /bin/bash banana
WORKDIR /app
# Stable Diffusioni Webui
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \
    git checkout baf6946e06249c5af9851c60171692c44ef633e0 &&\
    cd models/Stable-diffusion/ && \
    wget -O meinamix_meinaV10.safetensors --no-verbose --show-progress --progress=bar:force:noscroll ${MODEL_URL}

# Controlnet
WORKDIR /app/stable-diffusion-webui/extensions
RUN git clone https://github.com/Mikubill/sd-webui-controlnet.git && \
    cd sd-webui-controlnet && \
    git checkout 99408b9f4e514efdf33b19f3215ab661b989e209 &&\
    cd models && \
    wget -O control_v11p_sd15_lineart.pth --no-verbose --show-progress --progress=bar:force:noscroll ${CONTROL_URL}

WORKDIR /app/stable-diffusion-webui
RUN pip install tqdm requests
ADD prepare.py .
RUN python prepare.py --skip-torch-cuda-test --xformers
# ADD download.py download.py
# RUN python download.py --use-cpu=all

RUN mkdir -p extensions/banana/scripts
ADD script.py extensions/banana/scripts/banana.py
ADD app.py app.py
ADD server.py server.py

# CMD ["python", "server.py", "--xformers", "--disable-safe-unpickle", "--lowram", "--no-hashing", "--listen", "--port", "8000"]
