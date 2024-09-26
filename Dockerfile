FROM tensorflow/tensorflow:2.14.0-gpu
ENV TF_CPP_MIN_LOG_LEVEL=1
RUN apt-get update && \
  apt-get install -q -y \
  git \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/walline/prosub
WORKDIR /prosub
RUN pip install -r requirements.txt
