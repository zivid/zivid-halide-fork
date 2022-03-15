FROM ubuntu:bionic

ENV DEBIAN_FRONTEND=noninteractive
ARG PY310=https://github.com/indygreg/python-build-standalone/releases/download/20220227/cpython-3.10.2+20220227-x86_64-unknown-linux-gnu-install_only.tar.gz
ARG PY39=https://github.com/indygreg/python-build-standalone/releases/download/20220227/cpython-3.9.10+20220227-x86_64-unknown-linux-gnu-install_only.tar.gz

RUN apt-get update && apt-get upgrade --assume-yes
RUN apt-get install g++-8 ninja-build cmake wget unzip zip curl git -y
RUN mkdir /root/venv

WORKDIR /root/310
RUN wget $PY310 && tar xf cpython-3.10.2+20220227-x86_64-unknown-linux-gnu-install_only.tar.gz && rm cpython-3.10.2+20220227-x86_64-unknown-linux-gnu-install_only.tar.gz
RUN /root/310/python/bin/python3.10 -m venv /root/venv/310

WORKDIR /root/39
RUN wget $PY39 && tar xf cpython-3.9.10+20220227-x86_64-unknown-linux-gnu-install_only.tar.gz && rm cpython-3.9.10+20220227-x86_64-unknown-linux-gnu-install_only.tar.gz
RUN /root/39/python/bin/python3.9 -m venv /root/venv/39

WORKDIR /
ENV CC=gcc-8
ENV CXX=g++-8
