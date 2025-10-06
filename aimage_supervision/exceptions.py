"""
Custom exceptions for aimage-supervision backend

This module defines specific exception classes to provide better error handling
and appropriate HTTP status codes in API responses.
"""


class ExportError(Exception):
    """Base exception for export-related errors"""
    pass


class FileTooLargeError(ExportError):
    """Raised when a file is too large to process"""
    pass


class ContentNotFoundError(ExportError):
    """Raised when required content is not found for export"""
    pass


class InvalidPathError(ExportError):
    """Raised when S3 path is invalid or empty"""
    pass
