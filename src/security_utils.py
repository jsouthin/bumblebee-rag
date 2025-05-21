import os
from pathlib import Path
from typing import Union, Optional
import re
from urllib.parse import urlparse
import magic
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class SecurityValidationError(Exception):
    """Custom exception for security validation errors."""
    pass

def sanitize_filepath(filepath: Union[str, Path]) -> Path:
    """
    Sanitize and validate file path to prevent path traversal attacks.
    
    Args:
        filepath: The file path to sanitize
        
    Returns:
        Path: Sanitized Path object
        
    Raises:
        SecurityValidationError: If path validation fails
    """
    try:
        # Convert to Path object and resolve to absolute path
        path = Path(filepath).resolve()
        
        # Get the workspace root (assuming this is running in the workspace)
        workspace_root = Path.cwd().resolve()
        
        # Check if path is within workspace
        if not str(path).startswith(str(workspace_root)):
            raise SecurityValidationError("File path must be within workspace directory")
            
        # Check if path exists
        if not path.exists():
            raise SecurityValidationError(f"File does not exist: {path}")
            
        return path
        
    except Exception as e:
        raise SecurityValidationError(f"Invalid file path: {str(e)}")

def validate_file_type(filepath: Path, allowed_mime_types: Optional[list] = None) -> None:
    """
    Validate file type using magic numbers.
    
    Args:
        filepath: Path to the file
        allowed_mime_types: List of allowed MIME types. If None, accepts common document types.
        
    Raises:
        SecurityValidationError: If file type validation fails
    """
    if allowed_mime_types is None:
        allowed_mime_types = [
            'application/pdf',
            'text/plain',
            'text/csv',
            'application/json',
            'text/html',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ]
    
    try:
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(str(filepath))
        
        if file_type not in allowed_mime_types:
            raise SecurityValidationError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        raise SecurityValidationError(f"File type validation failed: {str(e)}")

def validate_url(url: str) -> str:
    """
    Validate and sanitize URL.
    
    Args:
        url: URL to validate
        
    Returns:
        str: Validated URL
        
    Raises:
        SecurityValidationError: If URL validation fails
    """
    try:
        # Parse URL
        parsed = urlparse(url)
        
        # Ensure scheme is http or https
        if parsed.scheme not in ['http', 'https']:
            raise SecurityValidationError("URL must use HTTP or HTTPS protocol")
            
        # Validate hostname
        if not parsed.netloc:
            raise SecurityValidationError("Invalid URL: missing hostname")
            
        # Optional: Add more validation (e.g., blacklist/whitelist domains)
        
        return url
        
    except Exception as e:
        raise SecurityValidationError(f"Invalid URL: {str(e)}")

def create_secure_session(
    timeout: int = 10,
    retries: int = 3,
    backoff_factor: float = 0.3
) -> requests.Session:
    """
    Create a requests Session with security settings.
    
    Args:
        timeout: Request timeout in seconds
        retries: Number of retries for failed requests
        backoff_factor: Backoff factor between retries
        
    Returns:
        requests.Session: Configured session object
    """
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    
    # Configure adapter with retry strategy
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set default timeout
    session.timeout = timeout
    
    # Ensure SSL verification is enabled
    session.verify = True
    
    return session 