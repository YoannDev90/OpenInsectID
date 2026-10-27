"""Tests for the MobileNetV4-style backbones and UIB blocks."""

import torch
import pytest

from src.models.backbones import (
    build_backbone,
    UIB,
    MobileNetV4,
    make_divisible,
    count_parameters,
    BACKBONE_TABLES,
)


def test_make_divisible():
    assert make_divisible(7) == 8
    assert make_divisible(8) == 8
    assert make_divisible(24) == 24
    assert make_divisible(100) % 8 == 0


def test_uib_shapes_stride1_and_stride2():
    x = torch.randn(2, 16, 32, 32)
    # stride 1, same channels -> residual path
    b1 = UIB(16, 16, stride=1, expansion=2.0, kernel=3)
    assert b1(x).shape == x.shape
    assert b1.use_skip is True
    # stride 2 -> downsample, no residual
    b2 = UIB(16, 32, stride=2, expansion=2.0, kernel=3)
    assert b2(x).shape == (2, 32, 16, 16)
    assert b2.use_skip is False


@pytest.mark.parametrize("name", ["mnv4_small", "mnv4_medium"])
def test_backbone_forward_shapes(name):
    bb = build_backbone(name)
    x = torch.randn(2, 3, 224, 224)
    out = bb(x)
    assert out.shape == (2, bb.out_channels)
    assert out.dim() == 2


def test_backbone_width_mult_changes_params():
    small = count_parameters(build_backbone("mnv4_small", width_mult=1.0))
    narrow = count_parameters(build_backbone("mnv4_small", width_mult=0.5))
    assert narrow < small


def test_backbone_stride_reduces_spatial():
    bb = build_backbone("mnv4_small")
    bb.eval()
    x1 = torch.randn(1, 3, 224, 224)
    x2 = torch.randn(1, 3, 112, 112)
    # Global pooling -> output dim independent of input size.
    assert bb(x1).shape == bb(x2).shape


def test_unknown_backbone_raises():
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_backbone("resnet50")


def test_timm_prefix_is_optional():
    # The timm seam should raise a clear ImportError only if timm is absent.
    try:
        import timm  # noqa: F401
        bb = build_backbone("timm:mobilenetv5_base")
        assert bb.out_channels > 0
    except ImportError:
        with pytest.raises(ImportError, match="timm"):
            build_backbone("timm:whatever")
