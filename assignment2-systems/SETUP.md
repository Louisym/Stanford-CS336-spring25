# Environment Setup Guide

## PyTorch Nightly with RTX 5090 (sm_120 support)

This project uses PyTorch nightly to support NVIDIA RTX 5090 GPU (CUDA compute capability sm_120).

### Why requirements.txt instead of pyproject.toml?

PyTorch nightly builds have special characteristics that make them incompatible with uv's standard dependency resolution:
- Non-standard version numbers (e.g., `2.10.0.dev20251013+cu128`)
- Dependencies with git hashes (e.g., `pytorch-triton==3.5.0+git7416ffcb`)
- Special index structure that requires `--index-strategy unsafe-best-match`

This is a known issue documented in [uv #10712](https://github.com/astral-sh/uv/issues/10712) and related issues.

### Installation

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Install dependencies (including PyTorch nightly)
uv pip install --index-strategy unsafe-best-match -r requirements.txt
```

### Running Scripts

**DO NOT use `uv run`** - it will reinstall an incompatible PyTorch version.

Instead, use the virtual environment directly:

```bash
# Option 1: Use .venv directly
.venv/bin/python your_script.py

# Option 2: Activate environment
source .venv/bin/activate
python your_script.py
```

### Verifying Installation

```bash
.venv/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'sm_120 support: {\"sm_120\" in torch.cuda.get_arch_list()}')"
```

Expected output:
```
PyTorch: 2.10.0.dev20251013+cu128
CUDA available: True
sm_120 support: True
```

### Running with nsys Profiling

```bash
~/bin/nsys profile -o result .venv/bin/python your_script.py
```

### Troubleshooting

**Problem**: `RuntimeError: CUDA error: no kernel image is available`

**Cause**: Wrong PyTorch version (not nightly) was installed

**Solution**: Reinstall with requirements.txt:
```bash
rm -rf .venv
uv venv
uv pip install --index-strategy unsafe-best-match -r requirements.txt
```

### References

- [uv Issue #10712](https://github.com/astral-sh/uv/issues/10712) - PyTorch for ROCm needs explicit triton installation
- [uv Issue #10693](https://github.com/astral-sh/uv/issues/10693) - PyTorch nightly inconsistency
- [PyTorch Nightly Builds](https://download.pytorch.org/whl/nightly/cu128/)
