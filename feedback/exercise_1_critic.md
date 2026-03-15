# Exercise 1 — Critic & Feedback

## What went well

- Conceptual understanding was solid throughout. You figured out *why* betas don't go to 1, why variance is preserved, why you train at all timesteps — usually with just one leading question. That's the hard part. It's there.
- You structured the code cleanly: functions separated, clear variable names, training loop logic was correct once debugged.
- You asked good questions. "Why sin and cos specifically?" and "Why not just pass t directly?" are exactly the questions a good engineer asks.
- You caught the `self.embedding` vs local variable issue yourself after a hint. Good reflex.

---

## What needs work

### 1. PyTorch API fluency — below where it should be

You needed prompting for: `torch.randn_like`, `torch.randint`, `torch.sqrt` vs `math.sqrt`, `F.relu` vs `nn.ReLU`, `torch.cat`, `[:, None]` broadcasting. These are not diffusion-specific — they are everyday PyTorch. You should not need to ask about these.

**Fix:** 30 minutes in a Python REPL, no notes, no help. Write tensors, manipulate them, break things, read error messages. Do this before Exercise 2.

### 2. Shape reasoning is reactive, not proactive

You wrote code and waited for it to break rather than tracing shapes in your head first. The `[batch]` × `[batch, 2]` broadcasting bug and the `T` variable collision were both things you could have caught by reading your own code carefully.

**Fix:** For every tensor operation you write, pause and ask: "what shape goes in, what shape comes out?" Write it as a comment if needed. Make this a habit before Exercise 2.

### 3. Scalar indexing vs slice indexing
You thought `a[:, 0]` gives `[10, 1]`. This is a fundamental distinction:
- `a[:, 0]` → scalar index → dimension is removed → `[10]`
- `a[:, 0:1]` → slice → dimension is kept → `[10, 1]`

You will hit this constantly in U-Net. Know it cold.

### 4. Variable naming discipline
You reused `T` for both total timesteps and batch timesteps. In a 100-line file this caused a silent bug. In a 500-line U-Net it will cost you an hour of debugging.

**Fix:** Name things precisely. `T` = total steps (constant). `t_batch` = per-sample timesteps (tensor). Never reuse.

### 5. Didn't verify Exercise 1 before building Exercise 2
You commented out the spiral visualization and jumped straight to the training loop. Always verify the previous piece works before building on it. The forward diffusion plot should have been confirmed working first.

---

## Bugs you made that you should have caught yourself

| Bug | How you would have caught it |
|-----|------------------------------|
| `math.sqrt` on tensor | Read the error: `TypeError: must be real number` |
| `model(x_t, t)` with integer `t` | Trace through `sinusoidal_embedding` — `t[:, None]` fails on a scalar |
| `T` variable collision | Read the training loop top to bottom before running |
| `alpha_bars[t]` shape mismatch | Ask: "t is [64], alpha_bars[t] is [64], x_0 is [64,2] — will `*` work?" |

---

## Verdict

The concepts are landing. The execution is sloppy. You're relying on errors and hints to catch things that careful reading would surface first. That gap is exactly what separates a researcher who can explain diffusion from an engineer who can ship it.

**Before Exercise 2:** finish the 20 tensor shape exercises. No moving on until shapes are automatic.
