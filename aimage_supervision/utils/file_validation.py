"""
File validation utilities for handling different file types including images, PSD, and AI files.
"""

from typing import Optional


def is_valid_image_or_design_file(filename: Optional[str], content_type: Optional[str]) -> bool:
    """
    Check if a file is a valid image, PSD, or AI file based on MIME type and extension.
    
    Args:
        filename: The filename with extension
        content_type: The MIME type of the file
    
    Returns:
        bool: True if the file is valid, False otherwise
    """
    # List of valid MIME types
    valid_mime_types = [
        'image/',  # All image types (image/jpeg, image/png, image/gif, etc.)
        'application/postscript',  # AI files (older EPS-based)
        'application/pdf',  # AI files (newer PDF-based)
        'application/x-photoshop',  # PSD files
        'application/psd',  # PSD files
        'image/vnd.adobe.photoshop',  # PSD files (official IANA)
        'application/illustrator',  # AI files
        'application/x-illustrator',  # AI files
    ]
    
    # Check MIME type
    if content_type:
        for valid_type in valid_mime_types:
            if content_type.startswith(valid_type):
                return True
    
    # Fallback to file extension checking
    if filename:
        ext = filename.lower().split('.')[-1]
        valid_extensions = [
            # Image formats
            'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'ico', 'tiff', 'tif',
            # Vector formats
            'svg', 'svgz',
            # Adobe formats
            'psd', 'psb',  # Photoshop
            'ai', 'eps',  # Illustrator
        ]
        if ext in valid_extensions:
            return True
    
    return False


def get_file_format_from_extension(filename: str) -> str:
    """
    Get the file format based on the file extension.
    
    Args:
        filename: The filename with extension
    
    Returns:
        str: The file format (e.g., 'JPEG', 'PNG', 'PSD', 'AI', 'SVG')
    """
    if not filename:
        return 'UNKNOWN'
    
    ext = filename.lower().split('.')[-1]
    
    format_mapping = {
        'jpg': 'JPEG',
        'jpeg': 'JPEG',
        'png': 'PNG',
        'gif': 'GIF',
        'bmp': 'BMP',
        'webp': 'WEBP',
        'ico': 'ICO',
        'tiff': 'TIFF',
        'tif': 'TIFF',
        'svg': 'SVG',
        'svgz': 'SVG',
        'psd': 'PSD',
        'psb': 'PSD',
        'ai': 'AI',
        'eps': 'EPS',
    }
    
    return format_mapping.get(ext, ext.upper())