#!/bin/bash

# 安装 Python 依赖
pip3 install virtualenv pypatch
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt