import torch


def sample_modality_states(batch_size: int, p_rgb: float, p_ir: float):
    if p_rgb + p_ir > 1.0:
        raise ValueError("p_rgb + p_ir must be <= 1.0")

    rand = torch.rand(batch_size)
    mask_drop_rgb = rand < p_rgb
    mask_drop_ir = (rand >= p_rgb) & (rand < p_rgb + p_ir)
    keep_both = ~(mask_drop_rgb | mask_drop_ir)
    return keep_both.float().mean().item(), mask_drop_rgb.float().mean().item(), mask_drop_ir.float().mean().item()


def assert_close(actual: float, expected: float, tol: float = 0.01):
    assert abs(actual - expected) <= tol, f"actual={actual:.4f}, expected={expected:.4f}, tol={tol:.4f}"


def test_modality_dropout_direct_semantics_distribution():
    batch_size = 10000
    cases = [
        (0.10, 0.10, 0.80, 0.10, 0.10),
        (0.20, 0.20, 0.60, 0.20, 0.20),
        (0.15, 0.10, 0.75, 0.15, 0.10),
        (0.00, 0.00, 1.00, 0.00, 0.00),
    ]

    for p_rgb, p_ir, exp_keep, exp_only_ir, exp_only_rgb in cases:
        keep_both, only_ir, only_rgb = sample_modality_states(batch_size, p_rgb, p_ir)
        assert_close(keep_both, exp_keep)
        assert_close(only_ir, exp_only_ir)
        assert_close(only_rgb, exp_only_rgb)


def test_modality_dropout_direct_semantics_invalid_probability():
    try:
        sample_modality_states(100, 0.6, 0.5)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when p_rgb + p_ir > 1.0")
