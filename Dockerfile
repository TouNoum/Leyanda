FROM tensorflow/tensorflow:latest-gpu-jupyter

WORKDIR /tf

COPY requirements.txt /tf/requirements.txt

RUN apt-get update && apt-get install -y git wget unzip ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r /tf/requirements.txt

EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
