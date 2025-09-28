#!/bin/bash
set -euo pipefail

# Usage: ./run_dev_env.sh [.env.dev]
# Builds Dockerfile.local and runs with --env-file

ENV_FILE=${1:-.env.dev}
IMAGE_NAME=aimage-generation-backend:dev
CONTAINER_NAME=aimage-generation-backend-dev
PORT=${PORT:-8000}

# Enable BuildKit for better caching
export DOCKER_BUILDKIT=1

if [ ! -f "$ENV_FILE" ]; then
  echo "Error: env file '$ENV_FILE' not found. Create it or pass a path explicitly."
  exit 1
fi

echo "Removing existing container if any..."
docker rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true

echo "Building image (${IMAGE_NAME}) using Dockerfile.local with BuildKit caching..."

# Check if the image exists by looking for it in the output
if docker images -q ${IMAGE_NAME} | grep -q .; then
  echo "Image ${IMAGE_NAME} exists. Building from cache..."
  docker build \
    --progress=plain \
    -t ${IMAGE_NAME} \
    --cache-from ${IMAGE_NAME} \
    -f Dockerfile.local .
else
  echo "Image ${IMAGE_NAME} does not exist. Building from scratch..."
  docker build \
    --progress=plain \
    -t ${IMAGE_NAME} \
    -f Dockerfile.local .
fi


echo "Starting container ${CONTAINER_NAME} with env file ${ENV_FILE}..."

# Determine if we're in development mode (default) or production mode
DEV_MODE=${DEV_MODE:-true}

if [ "$DEV_MODE" = "true" ]; then
  echo "Running in DEVELOPMENT mode with volume mounts for live code reloading..."
  docker run \
    --name ${CONTAINER_NAME} \
    --env-file "${ENV_FILE}" \
    -e PORT=${PORT} \
    -p ${PORT}:${PORT} \
    -v "$(pwd):/root/code" \
    -v aimage-generation-backend-pip-cache:/root/.cache/pip \
    --restart unless-stopped \
    ${IMAGE_NAME}
else
  echo "Running in PRODUCTION mode without volume mounts..."
  docker run \
    --name ${CONTAINER_NAME} \
    --env-file "${ENV_FILE}" \
    -e PORT=${PORT} \
    -p ${PORT}:${PORT} \
    --restart unless-stopped \
    ${IMAGE_NAME}
fi