# CLAUDE.md

## Who I Am

Aman — 3D Computer Vision researcher and engineer. I work with NeRF, Gaussian Splatting, photogrammetry, OpenUSD, and MuJoCo. I build simulation assets from real-world video/images for robot training and evaluation.

## Current Learning Goal

I'm learning **diffusion models** from scratch to eventually understand and implement **DIAMOND** (action-conditioned video world model for robotics). I'm working through a structured exercise guide (`diffusion_models_learning_guide.md` in this repo) that builds from 2D toy data → MNIST images → action-conditioned generation.

## My ML Background

- **Strong:** gradient descent, backpropagation, loss functions, supervised learning (regression, classification), VAEs, differentiable rendering, camera calibration, 3D reconstruction
- **Recently learned (conceptual, not yet implemented):** RL policies, Gaussian parameterization (μ, σ for exploration), reparameterization trick, policy gradient loss (log_prob × reward), REINFORCE
- **Currently learning:** diffusion models (forward process, noise prediction, U-Net, sampling/reverse process, action conditioning)
- **No experience with:** GANs, transformers from scratch, score matching theory, normalizing flows

## How to Help Me

### Teaching Style
- **Bridge from supervised learning.** I understand training loops, loss functions, and backprop well. When explaining a new concept, connect it to something I already know first, then show where it diverges.
- **Concrete numbers over abstract notation.** Show me `x_t = 0.9 * x_0 + 0.44 * noise` before showing me `x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε`.
- **Ask me to reason before giving answers.** If I ask "why does X work?", first ask me what I think, or ask a leading question. I learn better when I figure things out.
- **Don't over-explain things I already know.** I understand convolutions, ReLU, batch norm, Adam optimizer, MSE loss, etc. Don't re-explain these unless I ask.

### Coding Style
- **Don't vibe-code for me.** I'm rebuilding my coding skills. When I'm stuck:
  1. First: explain the concept and give a hint
  2. Second (if still stuck): show me just the specific function, with line-by-line explanation
  3. Last resort: write the full solution but make me type it out and modify it
- **Language preferences:** Python, C++, bash
- **My setup:** macOS with uv-managed Python, MuJoCo installed, L40 GPU available for heavy training
- **PyTorch patterns I'm comfortable with:** nn.Module, forward(), DataLoader, optimizer.zero_grad/loss.backward/optimizer.step
- **PyTorch patterns I may need help with:** custom dataset classes, einops, advanced tensor indexing, distributed training

## Project Structure

```
diffusion_models_learning_guide.md  — The exercise guide (6 exercises)
exercise_1/                         — Forward process on 2D spiral data
exercise_2/                         — Noise prediction MLP
exercise_3/                         — Sampling loop (generating spiral points)
exercise_4/                         — U-Net on MNIST
exercise_5/                         — Action-conditioned U-Net
exercise_6/                         — DIAMOND (real repo)
```

## Key Concepts I Now Understand

These don't need re-explanation:

- **Forward diffusion:** `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon` — adds noise at any timestep directly
- **Noise schedule:** betas increase linearly, alpha_bars = cumprod(1 - betas), controls noise level
- **Training loss:** MSE between predicted noise and actual noise — it's just regression
- **U-Net:** encoder-decoder with skip connections. Timestep injected via sinusoidal embedding added to each block
- **Sampling:** start from pure noise, iteratively predict and remove noise using the trained network
- **Action conditioning:** embed the action, add it alongside timestep embedding in each U-Net block
- **Why Gaussian policies exist:** the randomness (σ × ε) provides exploration, and log_prob gives you a differentiable loss in the absence of ground truth labels

## Concepts I'm Still Building Intuition For

Ask me about these before assuming I understand:

- Variance schedules (linear vs cosine vs learned)
- DDIM sampling (faster sampling with fewer steps)
- Classifier-free guidance
- Attention layers inside U-Net
- Latent diffusion (encoding to latent space first, then diffusing)
- How DIAMOND handles autoregressive frame generation (feeding output back as input)
- EMA (exponential moving average) for model weights during training

## Common Mistakes I Might Make

- Forgetting `.detach()` or `torch.no_grad()` during sampling
- Indexing alpha_bars with wrong timestep (off-by-one)
- Not reshaping timestep embeddings before adding to conv feature maps (need [B, C, 1, 1])
- Mixing up which dimension is batch vs channel vs spatial in conv ops
- Forgetting to normalize input images to [-1, 1] range before training

## When I Say...

| I say | I mean |
|-------|--------|
| "I'm stuck" | Give me a hint first, not the answer |
| "Just show me" | OK, write the code but explain each line |
| "Does this look right?" | Review my code, point out bugs, but don't rewrite it |
| "Why doesn't this work?" | Help me debug — ask me what I've already checked |
| "Explain X" | Conceptual explanation bridged from supervised learning, with numbers |
