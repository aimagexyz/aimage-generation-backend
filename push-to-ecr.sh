#!/bin/bash

# AWS ECR 配置
ECR_REGISTRY="590183972800.dkr.ecr.ap-northeast-1.amazonaws.com"
ECR_REPOSITORY="aimage-supervision-backend-dev"
IMAGE_TAG=${1:-latest}  # 默认使用 latest，也可以通过参数传入
FULL_IMAGE_NAME="${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"

echo "开始推送镜像到 AWS ECR..."
echo "目标仓库: ${FULL_IMAGE_NAME}"

# 检查 AWS CLI 是否已安装
if ! command -v aws &> /dev/null; then
    echo "错误: AWS CLI 未安装，请先安装 AWS CLI"
    exit 1
fi

# 检查 Docker 是否运行
if ! docker info &> /dev/null; then
    echo "错误: Docker 未运行，请先启动 Docker"
    exit 1
fi

echo "1. 登录到 AWS ECR..."
aws ecr get-login-password --region ap-northeast-1 | docker login --username AWS --password-stdin ${ECR_REGISTRY}

if [ $? -ne 0 ]; then
    echo "错误: ECR 登录失败"
    exit 1
fi

echo "2. 构建 Docker 镜像..."
docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} .

if [ $? -ne 0 ]; then
    echo "错误: Docker 镜像构建失败"
    exit 1
fi

echo "3. 标记镜像..."
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${FULL_IMAGE_NAME}

if [ $? -ne 0 ]; then
    echo "错误: 镜像标记失败"
    exit 1
fi

echo "4. 推送镜像到 ECR..."
docker push ${FULL_IMAGE_NAME}

if [ $? -ne 0 ]; then
    echo "错误: 镜像推送失败"
    exit 1
fi

echo "✅ 镜像推送成功！"
echo "镜像地址: ${FULL_IMAGE_NAME}"
echo ""
echo "使用方法:"
echo "  docker pull ${FULL_IMAGE_NAME}"
echo "  docker run -p 8000:8000 ${FULL_IMAGE_NAME}" 