import os
from typing import List, Optional

from google import genai
from pydantic import BaseModel, Field

# Model constants
QUALITY_MODEL = "gemini-2.5-pro"
SPEED_MODEL = "gemini-2.5-flash"


class GeminiDetectedElements(BaseModel):
    description: str = Field(
        ..., description="A comprehensive textual description of the key visual elements in the image.")
    characters: List[str] = Field(...,
                                  description="A list of detected characters in the image.")
    objects: List[str] = Field(...,
                               description="A list of detected objects in the image. Max 3 objects.")


async def detect_elements(image_path: Optional[str], character_candidates_list: Optional[List[str]] = None) -> GeminiDetectedElements:
    """Detects elements from an image using Gemini.
       Moved from endpoints/ai_review.py.
    """
    task_gemini_client = genai.Client(api_key=os.getenv(
        'GEMINI_API_KEY')) if os.getenv('GEMINI_API_KEY') else None
    if not task_gemini_client:
        raise ValueError(
            "GEMINI_API_KEY not set. Cannot detect bounding boxes.")

    if image_path is None:
        return GeminiDetectedElements(description="", characters=[], objects=[])

    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return GeminiDetectedElements(description="", characters=[], objects=[])

    contents = [
        "Your primary task is to provide a comprehensive textual description of the key visual elements in the image.",
        f"The following are the character candidates: {', '.join(character_candidates_list if character_candidates_list else [])}",
        genai.types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
        "List up the objects should be reviewed in the image. For example, 'Rengoku\\'s katana', 'Rengoku\\'s clothes', 'Rengoku\\'s face', etc."
    ]
    # Ensure this model name is current and appropriate for your use case
    model_name = SPEED_MODEL  # This is a fast operation, use speed model

    response = task_gemini_client.models.generate_content(
        model=model_name,
        contents=contents,
        config=genai.types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=GeminiDetectedElements,
            system_instruction="You are an expert image analysis AI. ",
            temperature=0.0,
        ),
    )
    # Accessing response.text and then validating with Pydantic is safer
    return response.parsed
