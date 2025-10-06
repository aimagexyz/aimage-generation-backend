import vertexai
from google import genai
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import MultiModalEmbeddingModel

from aimage_supervision.settings import (TEXT_EMBEDDING_MODEL, MULTIMODAL_EMBEDDING_MODEL, VERTEX_AI_PROJECT, VERTEX_AI_LOCATION)

import os
import base64
import json
from time import sleep

def init_google_cloud_credentials():
    """Initialize Google Cloud credentials from environment variables"""
        
    # Create credentials directory
    credentials_dir = './credentials'
    os.makedirs(credentials_dir, exist_ok=True)

    # Save credentials file
    credentials_file = credentials_dir + '/google_credentials.json'

    GOOGLE_CREDS = os.environ["GOOGLE_CREDS"]
    # Decode Base64 string
    google_creds_json = base64.b64decode(GOOGLE_CREDS).decode('utf-8')
    google_creds = json.loads(google_creds_json)

    with open(credentials_file, 'w') as f:
        json.dump(google_creds, f, indent=4)
    sleep(0.1)
        
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(credentials_file)
    return 
    
init_google_cloud_credentials()
vertexai.init()

try:
    gemini_client = genai.Client(vertexai=True, project=VERTEX_AI_PROJECT, location=VERTEX_AI_LOCATION)

except KeyError:    
    print('gemini_client init failed')
    exit(1) # Exit if API key is missing

try:
    text_embedding_model = TextEmbeddingModel.from_pretrained(TEXT_EMBEDDING_MODEL)
    multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained(MULTIMODAL_EMBEDDING_MODEL)
except Exception as e:
    print('text_embedding_model init failed')
    print('multimodal_embedding_model init failed')
    exit(1)