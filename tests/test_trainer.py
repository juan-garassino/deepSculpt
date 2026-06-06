#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from deepsculpt.core.training.pytorch_trainer import GANTrainer
from deepsculpt.core.training.base_trainer import TrainingConfig

print(f"GANTrainer class: {GANTrainer}")
print(f"Has train method: {hasattr(GANTrainer, 'train')}")
print(f"Train method: {GANTrainer.train if hasattr(GANTrainer, 'train') else 'NOT FOUND'}")

# Check MRO
print(f"\nMethod Resolution Order:")
for cls in GANTrainer.__mro__:
    print(f"  - {cls}")
    if hasattr(cls, 'train') and cls != object:
        print(f"    ✓ Has train method")
