"""Smoke tests for PHYSHADE-Net model zoo.

These run on CPU with tiny random tensors: they verify architecture
wiring (shapes, forward passes), not segmentation quality.
"""

import sys
from pathlib import Path

import pytest

# make src/ importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")
from physhade.models import AttentivePHYSHADENet, PHYSHADENet, UNet  # noqa: E402


@pytest.mark.parametrize("model_cls", [UNet, PHYSHADENet, AttentivePHYSHADENet])
def test_forward_output_shape(model_cls):
    """Each model maps a (B, C_in, H, W) tile to a (B, 1, H, W) mask."""
    model = model_cls(in_channels=4, out_channels=1)
    model.eval()
    x = torch.randn(2, 4, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 64, 64), f"{model_cls.__name__} output shape {y.shape}"


def test_unet_rgb_default():
    """Default UNet accepts 3-channel RGB input."""
    model = UNet()
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 32, 32)


def test_models_deterministic_eval():
    """In eval mode with fixed weights, repeated passes are identical."""
    model = PHYSHADENet(in_channels=4, out_channels=1)
    model.eval()
    x = torch.randn(1, 4, 32, 32)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)
    assert torch.allclose(y1, y2)
