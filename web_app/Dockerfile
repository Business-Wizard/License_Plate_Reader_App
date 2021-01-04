# FROM nvcr.io/nvidia/tensorflow:20.12-tf1-py3
# * for GPU based tensorflow training
FROM ubuntu:16.04

MAINTANER Joseph Wilson "jowi.legal.ia+dev@gmail.com"

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY web_app/requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY src /app/src
COPY . /app

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]