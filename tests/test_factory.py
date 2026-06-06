#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from deepsculpt.core.models.model_factory import PyTorchModelFactory

def test_factory():
    print("Testing PyTorchModelFactory...")
    
    # Create factory instance
    factory = PyTorchModelFactory()
    print(f"Factory created: {factory}")
    print(f"Factory type: {type(factory)}")
    print(f"Factory module: {factory.__class__.__module__}")
    
    # Check if method exists
    has_method = hasattr(factory, 'create_gan_generator')
    print(f"Has create_gan_generator method: {has_method}")
    
    if has_method:
        print("Method exists, trying to call it...")
        try:
            generator = factory.create_gan_generator(
                model_type='skip',
                void_dim=32,
                noise_dim=100,
                color_mode=0,
                sparse=False
            )
            print(f"Generator created successfully: {type(generator)}")
        except Exception as e:
            print(f"Error calling method: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Available methods:")
        methods = [m for m in dir(factory) if not m.startswith('_')]
        for method in methods:
            print(f"  - {method}")

if __name__ == "__main__":
    test_factory()