import torch

from ultralytics.nn.modules import C2f, CrossModalAlign, WaveletC2f
from ultralytics.nn.tasks import parse_model


def _forward_sequential(model, x):
    y = []
    for m in model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x)
    return x


def test_wavelet_c2f_shape_and_buffer():
    x = torch.randn(2, 64, 80, 64)
    wavelet_c2f = WaveletC2f(64, 64, n=1, shortcut=True)
    standard_c2f = C2f(64, 64, n=1, shortcut=True)
    assert wavelet_c2f(x).shape == standard_c2f(x).shape
    assert not wavelet_c2f.wavelet.filters.requires_grad


def test_wavelet_c2f_amp_forward():
    if not torch.cuda.is_available():
        return
    x = torch.randn(2, 64, 80, 64, device="cuda")
    wavelet_c2f = WaveletC2f(64, 64, n=1, shortcut=True).cuda()
    with torch.amp.autocast("cuda"):
        out = wavelet_c2f(x)
    assert out.shape == x.shape
    assert out.dtype == torch.float16


def test_cross_modal_align_shape_ir_identity():
    x_rgb = torch.randn(2, 64, 80, 64)
    x_ir = torch.randn(2, 64, 80, 64)
    align = CrossModalAlign(64)
    out_rgb, out_ir = align([x_rgb, x_ir])
    assert out_rgb.shape == x_rgb.shape
    assert out_ir.shape == x_ir.shape
    assert torch.equal(out_ir, x_ir)


def test_cross_modal_align_amp_forward():
    if not torch.cuda.is_available():
        return
    x_rgb = torch.randn(2, 64, 80, 64, device="cuda")
    x_ir = torch.randn(2, 64, 80, 64, device="cuda")
    align = CrossModalAlign(64).cuda()
    with torch.amp.autocast("cuda"):
        out_rgb, out_ir = align([x_rgb, x_ir])
    assert out_rgb.shape == x_rgb.shape
    assert out_ir.shape == x_ir.shape


def test_yaml_parse_cross_modal_align_chain():
    cfg = {
        "nc": 1,
        "depth_multiple": 1.0,
        "width_multiple": 1.0,
        "backbone": [
            [-1, 1, "IdentityInput", []],
            [0, 1, "ModalitySelector", [1]],
            [0, 1, "ModalitySelector", [2]],
            [1, 1, "Conv", [8, 3, 1]],
            [2, 1, "Conv", [8, 3, 1]],
            [[3, 4], 1, "CrossModalAlign", []],
            [-1, 1, "InceptionCoordAttnConcat", []],
            [-1, 1, "C2f", [8, True]],
        ],
        "head": [],
    }
    model, _ = parse_model(cfg, ch=6, verbose=False)
    x = torch.randn(1, 6, 64, 64)
    out = _forward_sequential(model, x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 8, 64, 64)


if __name__ == "__main__":
    test_wavelet_c2f_shape_and_buffer()
    test_wavelet_c2f_amp_forward()
    test_cross_modal_align_shape_ir_identity()
    test_cross_modal_align_amp_forward()
    test_yaml_parse_cross_modal_align_chain()
    print("test_fusion_upgrades passed")
