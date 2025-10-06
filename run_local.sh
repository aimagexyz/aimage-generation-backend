# Remove the existing container if it exists
docker rm -f aimage-generation-backend || true

# Remove the existing image if it exists
docker rmi -f aimage-generation-backend || true && docker build -t aimage-generation-backend -f Dockerfile.local .

# Run the container.
docker run --rm -p 8000:8000 -e PORT=8000 --name aimage-generation-backend aimage-generation-backend