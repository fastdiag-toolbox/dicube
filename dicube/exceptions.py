"""Exceptions module for DiCube.

This module defines the exception hierarchy used throughout the DiCube library.
All exceptions inherit from the base DicomCubeError class to allow for
easy catching of all DiCube-related exceptions.
"""

from typing import Optional, Dict, Any


class DicomCubeError(Exception):
    """Base exception class for all DicomCube-related errors.
    
    All other exceptions in the DiCube library inherit from this class,
    allowing applications to catch all DiCube-related exceptions with:
    
    ```python
    try:
        # DiCube operations
    except DicomCubeError:
        # Handle any DiCube error
    ```
    
    This enhanced base class supports contextual error information and
    helpful suggestions for resolving common issues.
    """
    
    def __init__(
        self, 
        message: str, 
        context: Optional[str] = None, 
        suggestion: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize a DicomCubeError with contextual information.
        
        Args:
            message (str): The primary error message.
            context (str, optional): Context about the operation that failed.
            suggestion (str, optional): Helpful suggestion for resolving the error.
            details (dict, optional): Additional structured error details.
        """
        self.context = context
        self.suggestion = suggestion
        self.details = details or {}
        
        formatted_message = self._format_message(message)
        super().__init__(formatted_message)
    
    def _format_message(self, message: str) -> str:
        """Format the error message with context and suggestions.
        
        Args:
            message (str): The base error message.
            
        Returns:
            str: The formatted error message.
        """
        parts = []
        
        # Add context if provided
        if self.context:
            parts.append(f"{self.context}: {message}")
        else:
            parts.append(message)
        
        # Add details if provided
        if self.details:
            detail_strs = [f"{k}={repr(v)}" for k, v in self.details.items()]
            parts.append(f"Details: {', '.join(detail_strs)}")
        
        # Add suggestion if provided
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        
        return "\n".join(parts)


class InvalidCubeFileError(DicomCubeError):
    """Raised when a file is not a valid DicomCube file.
    
    This exception is raised when attempting to load a file that is not
    in the expected DicomCube format. This could be due to incorrect 
    magic number, version mismatch, or a corrupted file structure.
    """
    
    def __init__(
        self, 
        message: str, 
        context: Optional[str] = None, 
        suggestion: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize InvalidCubeFileError with file-specific suggestions.
        
        Args:
            message (str): The primary error message.
            context (str, optional): Context about the operation that failed.
            suggestion (str, optional): Helpful suggestion for resolving the error.
            details (dict, optional): Additional structured error details.
        """
        if suggestion is None:
            suggestion = "Verify the file is a valid DicomCube file and not corrupted"
        
        super().__init__(message, context, suggestion, details)


class CodecError(DicomCubeError):
    """Raised when an error occurs in the encoding/decoding process.
    
    This exception is raised when there are problems with image compression
    or decompression, such as JPEG 2000 processing failures.
    """
    
    def __init__(
        self, 
        message: str, 
        context: Optional[str] = None, 
        suggestion: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize CodecError with codec-specific suggestions.
        
        Args:
            message (str): The primary error message.
            context (str, optional): Context about the operation that failed.
            suggestion (str, optional): Helpful suggestion for resolving the error.
            details (dict, optional): Additional structured error details.
        """
        if suggestion is None:
            suggestion = "Check image data format and codec compatibility"
        
        super().__init__(message, context, suggestion, details)


class MetaDataError(DicomCubeError):
    """Raised when metadata is missing or inconsistent.
    
    This exception is raised when critical metadata (DicomMeta, Space, etc.)
    is missing, corrupted, or inconsistent in a DicomCube file or operation.
    """
    
    def __init__(
        self, 
        message: str, 
        context: Optional[str] = None, 
        suggestion: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize MetaDataError with metadata-specific suggestions.
        
        Args:
            message (str): The primary error message.
            context (str, optional): Context about the operation that failed.
            suggestion (str, optional): Helpful suggestion for resolving the error.
            details (dict, optional): Additional structured error details.
        """
        if suggestion is None:
            suggestion = "Verify metadata completeness and consistency in the source data"
        
        super().__init__(message, context, suggestion, details)


class DataConsistencyError(DicomCubeError):
    """Raised when data arrays have consistency issues.
    
    This exception is raised when image data arrays have mismatched
    shapes, incompatible types, or other consistency-related issues.
    """
    
    def __init__(
        self, 
        message: str, 
        context: Optional[str] = None, 
        suggestion: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize DataConsistencyError with data-specific suggestions.
        
        Args:
            message (str): The primary error message.
            context (str, optional): Context about the operation that failed.
            suggestion (str, optional): Helpful suggestion for resolving the error.
            details (dict, optional): Additional structured error details.
        """
        if suggestion is None:
            suggestion = "Check data array shapes, types, and dimensional consistency"
        
        super().__init__(message, context, suggestion, details) 