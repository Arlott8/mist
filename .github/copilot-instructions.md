# Mist Project Instructions

## Project Overview
Mist is an image preprocessing tool designed to protect artwork from AI mimicry (e.g., Stable Diffusion) by adding adversarial watermarks. It is built upon the `stable-diffusion` codebase and uses `advertorch` for generating adversarial perturbations.

## Architecture & Core Components

- **Core Logic**: `mist_v3.py` is the main entry point for the watermarking process. It handles model loading, configuration, and the attack loop.
- **Web UI**: `mist-webui.py` provides a Gradio-based interface for the tool.
- **Adversarial Attack**:
  - `Masked_PGD.py`: Implements the Projected Gradient Descent (PGD) attack.
  - `models/diffusion/ddpmAttack.py`: A modified version of the standard DDPM implementation to support open gradient flow for end-to-end adversarial perturbation. **Crucial**: Use this instead of standard `ddpm.py` for attack-related logic.
- **Configuration**: Uses `omegaconf` with YAML files in `configs/`.
  - `configs/stable-diffusion/v1-inference-attack.yaml`: The specific config used for the attack pipeline.

## Developer Workflows

### Environment Setup
- The project relies on a Conda environment named `mist`.
- **Critical**: `Pillow` often needs a force reinstall after environment creation: `pip install --force-reinstall pillow`.
- **Checkpoints**: Requires Stable Diffusion v1.4 checkpoint at `models/ldm/stable-diffusion-v1/model.ckpt`.

### Running the Tool
- **CLI**: `python mist_v3.py` (Note: Check `mist_v3.py` for specific argument parsing, as it may differ from `mist_v2.py` documented in README).
- **WebUI**: `python mist-webui.py` launches the Gradio interface.

### Key Patterns & Conventions

- **Model Loading**: Use `load_model_from_config(config, ckpt)` in `mist_v3.py` to load the SDM model.
- **Loss Functions**:
  - `identity_loss`: Used for input to `advertorch`.
  - `target_model`: A virtual model wrapper that computes semantic and textural losses during the forward pass.
- **Image Processing**:
  - `mist_utils.py` contains helpers like `load_image_from_path`, `closing_resize`, and `load_mask`.
  - Images are typically processed as PIL images or numpy arrays and converted to tensors for the model.
- **Modifications to Upstream**:
  - Be aware that `ldm/` contains modified code from CompVis/stable-diffusion. Always check if you are editing a standard file or a Mist-specific variant (e.g., `ddpmAttack.py`).

## Common Issues
- **Gradient Flow**: If gradients are not propagating correctly during the attack, ensure you are using `ddpmAttack.py` components and not the frozen/standard ones.
- **Dependencies**: `clip` and `taming-transformers` are installed from source (git) in `src/`.

## Testing
- Use `utils/postprocess.py` to test robustness against transformations (crop, resize).
