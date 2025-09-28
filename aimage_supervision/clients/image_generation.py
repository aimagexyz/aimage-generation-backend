from google.genai import types
from typing import Optional
from aimage_supervision.clients.google import gemini_client

async def enhance_prompt(prompt: str, type: str) -> str:
    if type == 'translate':
        prompt = f'Translate the following prompt to English: {prompt}, only return the translated prompt, no other text.'
    elif type == 'enhance':
        prompt = f'Enhance the following prompt: {prompt}, only return the enhanced prompt, no other text.'
    elif type == 'translate_and_enhance':
        prompt = f'Translate the following prompt to English: {prompt}, enhance the translated prompt by adding details without change the original meaning. only return the enhanced prompt, no other text.'
    else:
        raise ValueError(f'Invalid type: {type}')

    response = gemini_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[prompt],
    )
    return response.text


async def generate_image_with_gemini(
    prompt: str, 
    number_of_images: int, 
    aspect_ratio: str,
    negative_prompt: Optional[str] = None,
):

    model = 'imagen-4.0-fast-generate-preview-06-06'

    response = gemini_client.models.generate_images(
        model=model,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=number_of_images,
            aspect_ratio=aspect_ratio,
            output_mime_type='image/jpeg',
            enhance_prompt=True,
            person_generation='ALLOW_ALL',
            language='ja',
        ),
    )

    return response.generated_images