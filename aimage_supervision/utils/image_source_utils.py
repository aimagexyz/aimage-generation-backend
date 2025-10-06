"""
Utility functions for extracting S3 image paths from different source types.
Follows DRY principle by centralizing image path extraction logic.
"""

from typing import Optional, Tuple
from uuid import UUID

from aimage_supervision.models import Character, GeneratedReference, Item


async def extract_image_path_from_source(
    source_type: str,
    source_id: UUID,
    image_url: str
) -> Optional[str]:
    """
    Extract S3 path from an image URL by looking up the source object.
    
    Args:
        source_type: Type of source object (character, item, generated_reference)
        source_id: UUID of the source object
        image_url: The image URL to find the corresponding S3 path for
        
    Returns:
        S3 path if found, None otherwise
        
    Raises:
        ValueError: If source_type is not supported
    """
    
    if source_type == "character":
        character = await Character.get_or_none(id=source_id)
        if not character:
            return None
            
        # Check main image
        if character.image_path:
            await character.fetch_image_url()
            if character.image_url() == image_url:
                return character.image_path
        
        # Check reference images
        if character.reference_images:
            await character.fetch_gallery_image_urls()
            gallery_urls = character.gallery_image_urls()
            for i, url in enumerate(gallery_urls):
                if url == image_url and i < len(character.reference_images):
                    return character.reference_images[i]
        
        # Check concept art images  
        if character.concept_art_images:
            await character.fetch_concept_art_image_urls()
            concept_urls = character.concept_art_image_urls()
            for i, url in enumerate(concept_urls):
                if url == image_url and i < len(character.concept_art_images):
                    return character.concept_art_images[i]
                    
    elif source_type == "item":
        item = await Item.get_or_none(id=source_id)
        if not item:
            return None
            
        # Check if this is the item's image
        await item.fetch_image_url()
        if item.image_url() == image_url:
            return item.s3_path
            
    elif source_type == "generated_reference":
        reference = await GeneratedReference.get_or_none(id=source_id)
        if not reference:
            return None
            
        # Check if this is the generated reference image
        await reference.fetch_image_url()
        if reference.image_url() == image_url:
            return reference.image_path
            
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")
    
    return None


def parse_s3_url_to_path(s3_url: str) -> Optional[str]:
    """
    Parse an S3 URL to extract the S3 path/key.
    
    Args:
        s3_url: S3 URL (presigned or s3:// format)
        
    Returns:
        S3 path/key if parseable, None otherwise
    """
    if not s3_url:
        return None
        
    # Handle s3:// format
    if s3_url.startswith('s3://'):
        parts = s3_url.split('/', 3)
        if len(parts) > 3:
            return parts[3]
    
    # Handle presigned URLs - extract path from URL
    # Example: https://bucket.s3.region.amazonaws.com/path/to/file.jpg?X-Amz-...
    if 'amazonaws.com/' in s3_url:
        try:
            # Find the path part after the domain
            domain_end = s3_url.find('.amazonaws.com/') + len('.amazonaws.com/')
            query_start = s3_url.find('?', domain_end)
            
            if query_start == -1:
                # No query parameters
                return s3_url[domain_end:]
            else:
                # Extract path before query parameters
                return s3_url[domain_end:query_start]
        except Exception:
            pass
    
    return None


async def get_source_info_from_url(image_url: str) -> Optional[Tuple[str, UUID, str]]:
    """
    Try to determine source information from an image URL by checking all possible sources.
    
    Args:
        image_url: The image URL to trace back to its source
        
    Returns:
        Tuple of (source_type, source_id, image_path) if found, None otherwise
    """
    
    # Try to parse S3 path from URL first
    image_path = parse_s3_url_to_path(image_url)
    if not image_path:
        return None
    
    # Search characters
    characters = await Character.all()
    for character in characters:
        if character.image_path == image_path:
            return ("character", character.id, image_path)
        if character.reference_images and image_path in character.reference_images:
            return ("character", character.id, image_path)
        if character.concept_art_images and image_path in character.concept_art_images:
            return ("character", character.id, image_path)
    
    # Search items
    items = await Item.all()
    for item in items:
        if item.s3_path == image_path:
            return ("item", item.id, image_path)
    
    # Search generated references
    references = await GeneratedReference.all()
    for reference in references:
        if reference.image_path == image_path:
            return ("generated_reference", reference.id, image_path)
    
    return None 