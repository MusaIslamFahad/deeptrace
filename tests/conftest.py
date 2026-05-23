"""
tests/conftest.py
Shared pytest fixtures and CI-safe mocking.
Ensures tests run in CI without a full GPU/ML environment.
"""

import sys
from unittest.mock import MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Mock heavy ML dependencies before any project code imports them.
# This lets the API, schema, and routing tests run in CI without
# installing torch, timm, or torchvision.
# ---------------------------------------------------------------------------

def _make_torch_mock():
    torch = MagicMock()

    # Tensor-like behaviour used in tests
    torch.zeros = MagicMock(return_value=MagicMock())
    torch.randn = MagicMock(return_value=MagicMock())
    torch.no_grad = MagicMock()
    torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    torch.device = MagicMock(return_value="cpu")

    # nn module
    torch.nn = MagicMock()
    torch.nn.Module = object        # so class inheritance works

    # softmax returns a mock that supports indexing
    prob_mock = MagicMock()
    prob_mock.__getitem__ = MagicMock(return_value=MagicMock())
    prob_mock.argmax = MagicMock(return_value=MagicMock())
    torch.nn.functional = MagicMock()
    torch.nn.functional.softmax = MagicMock(return_value=prob_mock)

    torch.cuda = MagicMock()
    torch.cuda.is_available = MagicMock(return_value=False)

    return torch


# Only mock if torch is not actually installed
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    sys.modules["torch"]                      = _make_torch_mock()
    sys.modules["torch.nn"]                   = sys.modules["torch"].nn
    sys.modules["torch.nn.functional"]        = sys.modules["torch"].nn.functional
    sys.modules["torch.optim"]                = MagicMock()
    sys.modules["torch.utils"]                = MagicMock()
    sys.modules["torch.utils.data"]           = MagicMock()
    sys.modules["torch.fft"]                  = MagicMock()
    sys.modules["timm"]                       = MagicMock()
    sys.modules["torchvision"]                = MagicMock()
    sys.modules["torchvision.transforms"]     = MagicMock()
    sys.modules["mlflow"]                     = MagicMock()
    sys.modules["mlflow.pytorch"]             = MagicMock()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def api_key():
    return "dev-key-123"


@pytest.fixture(scope="session")
def auth_headers(api_key):
    return {"X-API-Key": api_key}