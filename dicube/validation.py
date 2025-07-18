"""Validation utilities for DiCube library.

This module provides reusable validation functions that automatically raise
appropriate DicomCubeError exceptions with consistent error messages and context.
These utilities help minimize code duplication and provide clean error handling
throughout the DiCube library.
"""

import os
from typing import Any, Type, Union, Optional
import numpy as np

from .exceptions import (
    DicomCubeError,
    InvalidCubeFileError,
    DataConsistencyError,
    MetaDataError
)


def validate_not_none(
    value: Any, 
    name: str, 
    context: str = "",
    exception_class: Type[DicomCubeError] = DicomCubeError
) -> Any:
    """Validate that a value is not None.
    
    Args:
        value: The value to validate.
        name (str): Name of the parameter being validated.
        context (str): Context about the operation being performed.
        exception_class (Type[DicomCubeError]): Exception class to raise on failure.
        
    Returns:
        Any: The validated value if not None.
        
    Raises:
        DicomCubeError: If the value is None.
    """
    if value is None:
        raise exception_class(
            f"Parameter '{name}' cannot be None",
            context=context,
            details={"parameter": name, "value": value}
        )
    return value


def validate_file_exists(
    file_path: str, 
    context: str = "",
    exception_class: Type[DicomCubeError] = InvalidCubeFileError
) -> str:
    """Validate that a file exists.
    
    Args:
        file_path (str): Path to the file to validate.
        context (str): Context about the operation being performed.
        exception_class (Type[DicomCubeError]): Exception class to raise on failure.
        
    Returns:
        str: The validated file path.
        
    Raises:
        InvalidCubeFileError: If the file does not exist.
    """
    validate_not_none(file_path, "file_path", context, exception_class)
    
    if not os.path.exists(file_path):
        raise exception_class(
            f"File not found",
            context=context,
            details={"file_path": file_path},
            suggestion="Verify the file path is correct and the file exists"
        )
    
    if not os.path.isfile(file_path):
        raise exception_class(
            f"Path exists but is not a file",
            context=context,
            details={"file_path": file_path},
            suggestion="Ensure the path points to a file, not a directory"
        )
    
    return file_path


def validate_folder_exists(
    folder_path: str, 
    context: str = "",
    exception_class: Type[DicomCubeError] = InvalidCubeFileError
) -> str:
    """Validate that a folder exists.
    
    Args:
        folder_path (str): Path to the folder to validate.
        context (str): Context about the operation being performed.
        exception_class (Type[DicomCubeError]): Exception class to raise on failure.
        
    Returns:
        str: The validated folder path.
        
    Raises:
        InvalidCubeFileError: If the folder does not exist or is not a directory.
    """
    validate_not_none(folder_path, "folder_path", context, exception_class)
    
    if not os.path.exists(folder_path):
        raise exception_class(
            f"Folder not found",
            context=context,
            details={"folder_path": folder_path},
            suggestion="Verify the folder path is correct and the folder exists"
        )
    
    if not os.path.isdir(folder_path):
        raise exception_class(
            f"Path exists but is not a directory",
            context=context,
            details={"folder_path": folder_path},
            suggestion="Ensure the path points to a directory, not a file"
        )
    
    return folder_path


def validate_array_shape(
    array: np.ndarray, 
    expected_dims: Optional[int] = None,
    min_dims: Optional[int] = None,
    max_dims: Optional[int] = None,
    name: str = "array",
    context: str = "",
    exception_class: Type[DicomCubeError] = DataConsistencyError
) -> np.ndarray:
    """Validate numpy array shape and dimensions.
    
    Args:
        array (np.ndarray): The array to validate.
        expected_dims (int, optional): Expected number of dimensions.
        min_dims (int, optional): Minimum number of dimensions.
        max_dims (int, optional): Maximum number of dimensions.
        name (str): Name of the array parameter.
        context (str): Context about the operation being performed.
        exception_class (Type[DicomCubeError]): Exception class to raise on failure.
        
    Returns:
        np.ndarray: The validated array.
        
    Raises:
        DataConsistencyError: If array shape validation fails.
    """
    validate_not_none(array, name, context, exception_class)
    
    if not isinstance(array, np.ndarray):
        raise exception_class(
            f"Parameter '{name}' must be a numpy array",
            context=context,
            details={"parameter": name, "type": type(array).__name__},
            suggestion="Convert the data to a numpy array before passing"
        )
    
    actual_dims = array.ndim
    
    if expected_dims is not None and actual_dims != expected_dims:
        raise exception_class(
            f"Array '{name}' has incorrect dimensions",
            context=context,
            details={
                "parameter": name,
                "expected_dims": expected_dims,
                "actual_dims": actual_dims,
                "shape": array.shape
            },
            suggestion=f"Ensure the array has exactly {expected_dims} dimensions"
        )
    
    if min_dims is not None and actual_dims < min_dims:
        raise exception_class(
            f"Array '{name}' has too few dimensions",
            context=context,
            details={
                "parameter": name,
                "min_dims": min_dims,
                "actual_dims": actual_dims,
                "shape": array.shape
            },
            suggestion=f"Ensure the array has at least {min_dims} dimensions"
        )
    
    if max_dims is not None and actual_dims > max_dims:
        raise exception_class(
            f"Array '{name}' has too many dimensions",
            context=context,
            details={
                "parameter": name,
                "max_dims": max_dims,
                "actual_dims": actual_dims,
                "shape": array.shape
            },
            suggestion=f"Ensure the array has at most {max_dims} dimensions"
        )
    
    return array


def validate_parameter_type(
    value: Any, 
    expected_type: Union[Type, tuple], 
    name: str,
    context: str = "",
    exception_class: Type[DicomCubeError] = DataConsistencyError
) -> Any:
    """Validate parameter type.
    
    Args:
        value: The value to validate.
        expected_type (Type or tuple): Expected type or tuple of types.
        name (str): Name of the parameter being validated.
        context (str): Context about the operation being performed.
        exception_class (Type[DicomCubeError]): Exception class to raise on failure.
        
    Returns:
        Any: The validated value.
        
    Raises:
        DataConsistencyError: If the value is not of the expected type.
    """
    validate_not_none(value, name, context, exception_class)
    
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            type_names = [t.__name__ for t in expected_type]
            expected_str = " or ".join(type_names)
        else:
            expected_str = expected_type.__name__
        
        raise exception_class(
            f"Parameter '{name}' has incorrect type",
            context=context,
            details={
                "parameter": name,
                "expected_type": expected_str,
                "actual_type": type(value).__name__,
                "value": repr(value)
            },
            suggestion=f"Ensure '{name}' is of type {expected_str}"
        )
    
    return value


def validate_shape_consistency(
    array1: np.ndarray,
    array2: np.ndarray,
    name1: str = "array1",
    name2: str = "array2", 
    context: str = "",
    exception_class: Type[DicomCubeError] = DataConsistencyError
) -> tuple:
    """Validate that two arrays have consistent shapes.
    
    Args:
        array1 (np.ndarray): First array to compare.
        array2 (np.ndarray): Second array to compare.
        name1 (str): Name of the first array parameter.
        name2 (str): Name of the second array parameter.
        context (str): Context about the operation being performed.
        exception_class (Type[DicomCubeError]): Exception class to raise on failure.
        
    Returns:
        tuple: Both validated arrays.
        
    Raises:
        DataConsistencyError: If array shapes are inconsistent.
    """
    validate_array_shape(array1, name=name1, context=context, exception_class=exception_class)
    validate_array_shape(array2, name=name2, context=context, exception_class=exception_class)
    
    if array1.shape != array2.shape:
        raise exception_class(
            f"Arrays '{name1}' and '{name2}' have inconsistent shapes",
            context=context,
            details={
                name1 + "_shape": array1.shape,
                name2 + "_shape": array2.shape
            },
            suggestion="Ensure both arrays have the same dimensions and shape"
        )
    
    return array1, array2


def validate_string_not_empty(
    value: str,
    name: str,
    context: str = "",
    exception_class: Type[DicomCubeError] = DicomCubeError
) -> str:
    """Validate that a string is not None or empty.
    
    Args:
        value (str): The string value to validate.
        name (str): Name of the parameter being validated.
        context (str): Context about the operation being performed.
        exception_class (Type[DicomCubeError]): Exception class to raise on failure.
        
    Returns:
        str: The validated string.
        
    Raises:
        DicomCubeError: If the string is None or empty.
    """
    validate_not_none(value, name, context, exception_class)
    validate_parameter_type(value, str, name, context, exception_class)
    
    if not value.strip():
        raise exception_class(
            f"Parameter '{name}' cannot be empty",
            context=context,
            details={"parameter": name, "value": repr(value)},
            suggestion=f"Provide a non-empty value for '{name}'"
        )
    
    return value


def validate_numeric_range(
    value: Union[int, float],
    name: str,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    context: str = "",
    exception_class: Type[DicomCubeError] = DataConsistencyError
) -> Union[int, float]:
    """Validate that a numeric value is within a specified range.
    
    Args:
        value (int or float): The numeric value to validate.
        name (str): Name of the parameter being validated.
        min_value (int or float, optional): Minimum allowed value.
        max_value (int or float, optional): Maximum allowed value.
        context (str): Context about the operation being performed.
        exception_class (Type[DicomCubeError]): Exception class to raise on failure.
        
    Returns:
        int or float: The validated value.
        
    Raises:
        DataConsistencyError: If the value is outside the specified range.
    """
    validate_not_none(value, name, context, exception_class)
    validate_parameter_type(value, (int, float), name, context, exception_class)
    
    if min_value is not None and value < min_value:
        raise exception_class(
            f"Parameter '{name}' is below minimum value",
            context=context,
            details={
                "parameter": name,
                "value": value,
                "min_value": min_value
            },
            suggestion=f"Ensure '{name}' is at least {min_value}"
        )
    
    if max_value is not None and value > max_value:
        raise exception_class(
            f"Parameter '{name}' exceeds maximum value",
            context=context,
            details={
                "parameter": name,
                "value": value,
                "max_value": max_value
            },
            suggestion=f"Ensure '{name}' is at most {max_value}"
        )
    
    return value