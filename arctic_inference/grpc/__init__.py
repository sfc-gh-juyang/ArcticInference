"""
Arctic Inference gRPC module.

This module provides gRPC server and client implementations for the vLLM AsyncLLMEngine.
"""

# Use conditional imports to avoid circular imports when generating gRPC files
import importlib.util
import sys
from pathlib import Path

__all__ = [
    'InferenceServer',
    'InferenceServicer',
    'InferenceClient',
]

# Only import the classes if the modules are available
# This prevents circular imports when generating the gRPC code
def _import_if_available():
    try:
        # Check if the generated files exist
        if (Path(__file__).parent / "inference_pb2.py").exists():
            from arctic_inference.grpc.server import InferenceServer, InferenceServicer
            from arctic_inference.grpc.client import InferenceClient
            
            # Add the classes to the module namespace
            sys.modules[__name__].InferenceServer = InferenceServer
            sys.modules[__name__].InferenceServicer = InferenceServicer
            sys.modules[__name__].InferenceClient = InferenceClient
    except ImportError:
        # The generated files don't exist yet or there's another import issue
        pass

# Try to import the classes
_import_if_available() 