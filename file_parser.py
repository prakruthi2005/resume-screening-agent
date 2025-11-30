import os
from typing import List

def get_supported_formats() -> List[str]:
    """Get list of supported file formats"""
    return ['.pdf', '.docx', '.txt']

def validate_file_format(filename: str) -> bool:
    """Validate if file format is supported"""
    file_ext = os.path.splitext(filename)[1].lower()
    return file_ext in get_supported_formats()

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)