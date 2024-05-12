FROM pytorch/pytorch:latest
RUN apt-get update && apt-get install -y vim
RUN pip install -r requirements.txt
