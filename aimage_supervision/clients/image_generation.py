from typing import List, Optional

from google.genai import types

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
    return response.text or ""


async def generate_image_with_gemini(
    prompt: str, 
    number_of_images: int, 
    aspect_ratio: str,
    negative_prompt: Optional[str] = None,
):

    model = 'imagen-4.0-fast-generate-001'

    response = gemini_client.models.generate_images(
        model=model,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=number_of_images,
            aspect_ratio=aspect_ratio,
            output_mime_type='image/jpeg',
            enhance_prompt=True,
            # Use defaults for person_generation and language for widest compatibility
        ),
    )

    return response.generated_images


async def generate_image_with_gemini_from_images(
    prompt: str,
    input_images: List[bytes],
    aspect_ratio: str,
) -> Optional[bytes]:
    """Generate image using the gemini-2.5-flash-image-preview model with reference images.

    Returns a list of raw image bytes.
    """

    model = 'gemini-2.5-flash-image-preview'

    contents: list = []
    if aspect_ratio:
        prompt += f"The image should be in a {aspect_ratio} format."
    # Add text prompt first
    if prompt:
        contents.append(prompt)
    # Attach all reference images as inline parts
    for img_bytes in input_images:
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type='image/png'))

    response = gemini_client.models.generate_content(
        model=model,
        contents=contents,
    )

    for candidate in getattr(response, 'candidates', []) or []:
        content = getattr(candidate, 'content', None)
        if not content:
            continue
        for part in getattr(content, 'parts', []) or []:
            inline = getattr(part, 'inline_data', None)
            if inline and getattr(inline, 'data', None):
                return inline.data
