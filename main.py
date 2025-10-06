import uvicorn

if __name__ == '__main__':
    # Changed host to 0.0.0.0 to accept requests from all IPs
    uvicorn.run('aimage_supervision.app:app', host='0.0.0.0', port=8000)
