import torch


def is_ascend_npu():
    """Return whether the active PyTorch runtime exposes an Ascend NPU."""
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except (ImportError, AttributeError):
        return False
