#!/bin/bash

# AWS ECR 配置
ECR_REGISTRY="590183972800.dkr.ecr.ap-northeast-1.amazonaws.com"
ECR_REPOSITORY="aimage-supervision-backend"
IMAGE_TAG=${1:-latest}

echo "开始生产环境部署..."
echo "使用镜像: ${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"

# 检查 AWS CLI 是否已安装
if ! command -v aws &> /dev/null; then
    echo "错误: AWS CLI 未安装，请先安装 AWS CLI"
    exit 1
fi

# 登录到 ECR
echo "1. 登录到 AWS ECR..."
aws ecr get-login-password --region ap-northeast-1 | docker login --username AWS --password-stdin ${ECR_REGISTRY}

if [ $? -ne 0 ]; then
    echo "错误: ECR 登录失败"
    exit 1
fi

# 如果指定了特定标签，更新 docker-compose.prod.yml
if [ "$IMAGE_TAG" != "latest" ]; then
    echo "2. 更新镜像标签到 ${IMAGE_TAG}..."
    sed -i.bak "s/:latest/:${IMAGE_TAG}/g" docker-compose.prod.yml
fi

# 停止现有服务
echo "3. 停止现有服务..."
docker-compose -f docker-compose.prod.yml down

# 拉取最新镜像
echo "4. 拉取最新镜像..."
docker-compose -f docker-compose.prod.yml pull

# 启动服务
echo "5. 启动生产服务..."
docker-compose -f docker-compose.prod.yml up -d

# 等待服务启动
echo "6. 等待服务启动..."
sleep 15

# 检查服务状态
echo "7. 检查服务状态..."
docker-compose -f docker-compose.prod.yml ps

# 显示健康检查状态
echo "8. 检查应用健康状态..."
for i in {1..10}; do
    if curl -f http://localhost:8000/api/v1/health-check &> /dev/null; then
        echo "✅ 应用健康检查通过！"
        break
    else
        echo "等待应用启动... (${i}/10)"
        sleep 5
    fi
    
    if [ $i -eq 10 ]; then
        echo "⚠️  健康检查超时，请检查应用日志"
        docker-compose -f docker-compose.prod.yml logs --tail=20
    fi
done

# 显示部署信息
echo ""
echo "🚀 生产环境部署完成！"
echo "应用地址: http://localhost:8000"
echo ""
echo "常用命令:"
echo "  查看日志: docker-compose -f docker-compose.prod.yml logs -f"
echo "  停止服务: docker-compose -f docker-compose.prod.yml down"
echo "  重启服务: docker-compose -f docker-compose.prod.yml restart"
echo "  查看状态: docker-compose -f docker-compose.prod.yml ps" 