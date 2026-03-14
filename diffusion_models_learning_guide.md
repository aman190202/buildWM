# Learning Diffusion Models from Scratch — A Coding Guide

**Who this is for:** Someone who understands backprop, loss functions, supervised learning, and VAEs, but has never implemented a diffusion model. You should be comfortable with PyTorch basics (tensors, nn.Module, optimizers) even if you're rusty.

**How to use this:** Work through each exercise in order with Claude Code. Don't vibe-code — type the code yourself, ask Claude Code to *explain* when you're stuck rather than *write it for you*. Each exercise builds on the previous one.

**Setup:** Python 3.11+, PyTorch 2.x, matplotlib, tqdm. A MacBook CPU is fine for exercises 1-3. GPU (your L40) needed for exercises 4-5.

```bash
pip install torch torchvision matplotlib tqdm
```

---

## Exercise 1: The Forward Process (Destroying Data)

**Goal:** Understand how clean data gets progressively noised. No neural network yet.

**The math you need:**

The forward process adds noise to data according to a schedule:

```
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

Where:
- `x_0` is your clean data
- `epsilon ~ Normal(0, 1)` is random noise
- `alpha_bar_t` is a number between 1.0 (t=0, no noise) and ~0.0 (t=T, pure noise)
- `alpha_bar_t = product(alpha_1 * alpha_2 * ... * alpha_t)`
- `alpha_t = 1 - beta_t`
- `beta_t` is a small number that increases linearly from 0.0001 to 0.02

**What to implement:**

1. Create a 2D dataset: 1000 points arranged in a spiral or circle pattern
   ```python
   # Hint: use parametric equations
   theta = torch.linspace(0, 4 * math.pi, 1000)
   x = theta * torch.cos(theta)
   y = theta * torch.sin(theta)
   data = torch.stack([x, y], dim=1)  # shape [1000, 2]
   ```

2. Implement the noise schedule:
   ```python
   def get_noise_schedule(T=1000):
       """Return alpha_bar for each timestep 0..T-1"""
       betas = torch.linspace(0.0001, 0.02, T)
       alphas = 1.0 - betas
       alpha_bars = torch.cumprod(alphas, dim=0)
       return alpha_bars
   ```

3. Implement the forward noising function:
   ```python
   def forward_diffusion(x_0, t, alpha_bars):
       """
       Add noise to x_0 at timestep t.
       Returns: x_t (noised data), epsilon (the noise that was added)
       """
       # YOUR CODE HERE
       # Use the formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
       pass
   ```

4. Visualize: plot the spiral at t=0, t=250, t=500, t=750, t=999
   - You should see the spiral gradually dissolve into random noise

**Verify:**
- At t=0, the plot should look exactly like your original spiral
- At t=999, the plot should look like random Gaussian noise (no structure)
- `alpha_bars[0]` should be close to 1.0
- `alpha_bars[-1]` should be close to 0.0

**What to understand before moving on:**
- Why does `sqrt(alpha_bar)` multiply the signal and `sqrt(1 - alpha_bar)` multiply the noise?
  (Hint: it keeps the variance of x_t constant at 1.0 — if you add noise without scaling down the signal, the values would explode)
- Why is this formula useful? Because you can jump to ANY timestep directly — you don't need to noise step by step

---

## Exercise 2: The Noise Prediction Network

**Goal:** Build and train a tiny neural network that predicts what noise was added to data.

**The math:**

Training loss is MSE between predicted and actual noise:

```
L = || epsilon_predicted - epsilon_actual ||^2
```

That's it. Same as regression.

**What to implement:**

1. Build a small MLP (NOT a U-Net yet — we're working with 2D points, not images):
   ```python
   class NoisePredictor(nn.Module):
       """
       Takes: noisy 2D point x_t (2 dims) + timestep t (1 dim, embedded)
       Returns: predicted noise epsilon (2 dims)
       """
       def __init__(self, hidden_dim=128):
           super().__init__()
           # Timestep embedding: integer t -> vector
           # Use sinusoidal embedding (like positional encoding in transformers)
           # OR just a learned embedding table
           
           # MLP: takes [x_t (2) + t_embedding (32)] -> hidden -> hidden -> noise (2)
           # YOUR CODE HERE
           pass
       
       def forward(self, x_t, t):
           # YOUR CODE HERE
           pass
   ```

2. Implement the timestep embedding:
   ```python
   def sinusoidal_embedding(t, dim=32):
       """
       Convert integer timestep to a vector.
       Same idea as positional encoding in transformers.
       t: tensor of shape [batch_size]
       returns: tensor of shape [batch_size, dim]
       """
       # YOUR CODE HERE
       # Hint: use sin and cos at different frequencies
       # half_dim = dim // 2
       # freqs = exp(-log(10000) * arange(half_dim) / half_dim)
       # args = t[:, None] * freqs[None, :]
       # return cat([sin(args), cos(args)], dim=-1)
       pass
   ```

3. Implement the training loop:
   ```python
   for epoch in range(num_epochs):
       # 1. Sample a batch of clean data points x_0
       # 2. Sample random timesteps t for each point
       # 3. Sample random noise epsilon
       # 4. Create noisy data: x_t = forward_diffusion(x_0, t, alpha_bars)
       # 5. Predict noise: eps_pred = model(x_t, t)
       # 6. Loss = MSE(eps_pred, epsilon)
       # 7. Backprop, optimizer step
       pass
   ```

**Verify:**
- Loss should decrease steadily over training
- Start with a simple test: if t=999 (pure noise), the network can't predict anything — loss should stay high for those timesteps
- If t=1 (almost clean), the noise is tiny and structured — loss should be low

**Key insight:** Notice how this is just supervised learning. You KNOW the ground truth noise (you generated it in step 3). The diffusion framework turns the generative modeling problem into a regression problem.

---

## Exercise 3: Sampling (Generating New Data)

**Goal:** Use your trained network to generate new spiral points from pure noise.

**The math (simplified DDPM sampling):**

```
Start with x_T ~ Normal(0, I)

For t = T-1, T-2, ..., 0:
    eps_pred = model(x_t, t)
    
    x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_pred)
              + sqrt(beta_t) * z
    
    where z ~ Normal(0, I) if t > 0, else z = 0
```

Don't memorize this formula. Understand what it does:
- `eps_pred`: the network's guess of what noise is in x_t
- The big fraction: removes the predicted noise and rescales
- `sqrt(beta_t) * z`: adds a tiny bit of fresh noise (keeps the process stochastic)
- At the last step (t=0), you don't add noise — you want a clean output

**What to implement:**

1. The sampling loop:
   ```python
   @torch.no_grad()
   def sample(model, alpha_bars, T=1000, num_samples=500):
       """Generate new 2D points from pure noise."""
       betas = # ... reconstruct from alpha_bars
       alphas = 1.0 - betas
       
       # Start from pure noise
       x = torch.randn(num_samples, 2)
       
       for t in reversed(range(T)):
           t_batch = torch.full((num_samples,), t, dtype=torch.long)
           eps_pred = model(x, t_batch)
           
           # YOUR CODE: implement the update formula above
           # Be careful with the indexing — alpha_bars[t], betas[t], etc.
           
       return x
   ```

2. Generate 500 points and plot them alongside the original spiral

**Verify:**
- The generated points should roughly form a spiral shape
- They won't be perfect — this is a tiny network on simple data
- If you get random noise, your sampling loop probably has a bug in the formula
- If you get a blob (not a spiral), training may need more epochs

**Checkpoint:** At this point you understand the complete diffusion pipeline:
- Forward: add noise (easy, just math)
- Train: predict noise (supervised regression)
- Sample: iteratively remove predicted noise (apply the trained network T times)

---

## Exercise 4: Image Diffusion with a U-Net

**Goal:** Scale up to actual images. Move to your L40 GPU for this.

**Dataset:** MNIST (28×28 grayscale) — small enough to train in ~30 minutes.

**What changes from Exercise 2:**
- Data goes from 2D points to 28×28 images
- Network goes from MLP to U-Net (because convolutions are needed for spatial data)
- Everything else (forward process, loss function, sampling loop) stays THE SAME

**What to implement:**

1. A minimal U-Net:
   ```python
   class SimpleUNet(nn.Module):
       """
       Encoder:  28×28 → 14×14 → 7×7
       Decoder:  7×7 → 14×14 → 28×28
       Skip connections at each level
       Timestep embedding added to each block
       """
       def __init__(self, time_emb_dim=32):
           super().__init__()
           
           # Time embedding MLP
           self.time_mlp = nn.Sequential(
               nn.Linear(time_emb_dim, 64),
               nn.ReLU(),
               nn.Linear(64, 64),
           )
           
           # Encoder (downsampling)
           # Block 1: 1 channel → 32 channels, 28×28 → 14×14
           # Block 2: 32 channels → 64 channels, 14×14 → 7×7
           
           # Bottleneck: 64 channels → 64 channels at 7×7
           
           # Decoder (upsampling)
           # Block 3: 64+64 channels → 32 channels, 7×7 → 14×14
           #          (64 from upsampled + 64 from skip connection = 128 input channels)
           # Block 4: 32+32 channels → 1 channel, 14×14 → 28×28
           
           # YOUR CODE HERE
           pass
       
       def forward(self, x, t):
           """
           x: [batch, 1, 28, 28] noisy image
           t: [batch] timestep integers
           returns: [batch, 1, 28, 28] predicted noise
           """
           # 1. Embed timestep
           t_emb = sinusoidal_embedding(t)
           t_emb = self.time_mlp(t_emb)
           
           # 2. Encoder path (save outputs for skip connections)
           # 3. Bottleneck
           # 4. Decoder path (concatenate skip connections)
           # YOUR CODE HERE
           pass
   ```

   Implementation hints:
   - For downsampling: `nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)`
   - For upsampling: `nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)`
   - To add timestep embedding to a conv block: reshape t_emb to [batch, channels, 1, 1] and add it to the feature map
   - Skip connection: `torch.cat([upsampled, encoder_output], dim=1)` along channel dimension

2. Training loop (nearly identical to Exercise 2):
   ```python
   dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                         transform=transforms.ToTensor())
   dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
   
   model = SimpleUNet().cuda()
   optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
   
   for epoch in range(20):
       for batch, _ in dataloader:
           batch = batch.cuda()  # [128, 1, 28, 28]
           
           # Same 6 steps as Exercise 2, just with images instead of 2D points
           # YOUR CODE HERE
   ```

3. Sample and visualize generated digits

**Verify:**
- After 5 epochs: blurry but recognizable digit-like shapes
- After 20 epochs: clear digits (not perfect, but obviously digits)
- If you get gray blobs: check that you're adding timestep embedding correctly
- If you get noise: check your sampling loop formula (most common bug)

---

## Exercise 5: Action-Conditioned Diffusion (Mini-DIAMOND)

**Goal:** Condition generation on an action. This is the bridge to world models.

**Dataset:** Create a toy dataset where the "action" determines the output.
For example: MNIST digit with a rotation action.

```python
# Generate training data:
# (input_digit, action=rotation_angle) → output_digit (rotated)
# action=0: no rotation
# action=1: 90 degrees
# action=2: 180 degrees
# action=3: 270 degrees
```

**What changes from Exercise 4:**
- The U-Net now takes a THIRD input: the action
- Action gets its own embedding (small MLP, just like timestep)
- Both embeddings get added to each U-Net block

```python
class ActionConditionedUNet(nn.Module):
    def __init__(self, num_actions=4, time_emb_dim=32, action_emb_dim=32):
        super().__init__()
        
        self.time_mlp = nn.Sequential(...)   # same as before
        self.action_mlp = nn.Sequential(     # NEW
            nn.Embedding(num_actions, action_emb_dim),
            nn.Linear(action_emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        
        # ... same U-Net architecture ...
    
    def forward(self, x, t, action):
        t_emb = self.time_mlp(sinusoidal_embedding(t))
        a_emb = self.action_mlp(action)     # NEW
        
        # In each block, add BOTH embeddings:
        # features = features + t_emb + a_emb
        pass
```

**Training:**
```python
for input_img, action, target_img in dataloader:
    t = torch.randint(0, T, (batch_size,))
    eps = torch.randn_like(target_img)
    noisy_target = forward_diffusion(target_img, t, alpha_bars)
    
    eps_pred = model(noisy_target, t, action)  # action is now an input
    loss = F.mse_loss(eps_pred, eps)
```

**Sampling:**
```python
# "Show me what digit 7 looks like rotated 90 degrees"
x = torch.randn(1, 1, 28, 28).cuda()
action = torch.tensor([1]).cuda()  # 1 = 90 degrees

for t in reversed(range(T)):
    eps_pred = model(x, t, action)  # pass action at every denoising step
    x = denoise_step(x, eps_pred, t)
```

**Verify:**
- Given the same input digit + different actions → different rotations
- Given the same action + different input digits → consistent rotation angle
- This proves the model learned to condition on the action

**Why this matters:** Replace "rotated MNIST digit" with "next Atari frame" and "rotation angle" with "joystick input" and you have DIAMOND. The architecture is the same — just larger, with the previous game frame as additional context.

---

## Exercise 6 (Stretch Goal): Read and Run DIAMOND

Once exercises 1-5 are solid, you're ready for the real thing.

```bash
git clone https://github.com/eloialonso/diamond.git
cd diamond
pip install -r requirements.txt
```

1. **Play pretrained models first** — get intuition for what the model does:
   ```bash
   python src/play.py --pretrained
   ```

2. **Read the code** with your new understanding. Map each part to your exercises:
   - `src/models/diffusion.py` → your forward process and sampling loop
   - `src/models/unet.py` → your U-Net but bigger, with attention layers
   - Action conditioning → same pattern as Exercise 5
   - The "world model" part: previous frame is concatenated with x_t as input channels

3. **Train on a new Atari game** on your L40:
   ```bash
   python src/main.py env.train.id=BreakoutNoFrameskip-v4
   ```
   Expected: ~12-18 hours on L40, playable results

---

## Quick Reference: The Three Key Formulas

**1. Forward process (add noise):**
```
x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 - ᾱ_t) · ε,    ε ~ N(0, I)
```

**2. Training loss (predict noise):**
```
L = || ε_θ(x_t, t) - ε ||²
```

**3. Reverse step (remove noise):**
```
x_{t-1} = (1/√α_t) · (x_t - (β_t/√(1-ᾱ_t)) · ε_θ(x_t, t)) + √β_t · z
```

Everything else is architecture choices and engineering.

---

## Common Bugs and Fixes

| Symptom | Likely cause |
|---------|-------------|
| Generated samples are pure noise | Sampling loop formula is wrong — check signs and indexing |
| Generated samples are gray blobs | Timestep embedding not working — network doesn't know noise level |
| Loss doesn't decrease | Learning rate too high, or bug in forward diffusion |
| Loss decreases but samples are bad | Usually a sampling bug, not a training bug — double check the reverse formula |
| Samples all look the same | Network collapsed — reduce learning rate, check batch norm usage |
| Action conditioning has no effect | Action embedding too small, or not being added to enough layers |

---

## How to Use This with Claude Code

For each exercise, tell Claude Code:

> "I'm working on Exercise N of my diffusion model guide. Here's what I've implemented so far: [paste code]. I'm stuck on [specific part]. Don't write it for me — explain the concept and give me a hint so I can write it myself."

If you're truly stuck after trying:

> "I've tried X and Y but getting Z. Show me just the specific function I'm struggling with and explain each line."

The goal is to rebuild your coding muscles, not generate code. The exercises are ordered so each one is a small step from the previous — if an exercise feels like a huge jump, you skipped understanding something in the previous one.
