#!/bin/bash
source venv/bin/activate
aerich upgrade
uvicorn aimage_supervision.app:app --host 0.0.0.0 --port 8000
