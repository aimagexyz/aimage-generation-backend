import uvicorn

if __name__ == '__main__':
    # Local development
    uvicorn.run(
        'aimage_supervision.app:app',
        host='127.0.0.1',
        port=8000,
        reload=True,
    )
