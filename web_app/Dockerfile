# FROM nvcr.io/nvidia/tensorflow:20.12-tf1-py3
# * for GPU based tensorflow training
FROM ubuntu:16.04

LABEL "Author"="Joseph_Wilson"

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

COPY ./src /app/src
COPY . /app

ENV PYTHONPATH "${PYTHONPATH}:/app/src"
ENV GOOGLE_APPLICATION_CREDENTIALS "web_app/credentials/radiant-rookery-300603-6c38b0481439.json"

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 31000

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]