"""
Pytest configuration for DeepSculpt (PyTorch v2).

The previous conftest imported TensorFlow (archived stack) at module level,
which made every pytest invocation grind through a TF import — or fail where
TF isn't installed. This one is torch-only and keeps tests CPU + deterministic.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import pytest
import torch

# Repo root on sys.path so `import deepsculpt` works without installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session", autouse=True)
def force_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@pytest.fixture(scope="session", autouse=True)
def fixed_seeds():
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def tmp_dir():
    path = tempfile.mkdtemp(prefix="deepsculpt-test-")
    yield path
    shutil.rmtree(path, ignore_errors=True)
