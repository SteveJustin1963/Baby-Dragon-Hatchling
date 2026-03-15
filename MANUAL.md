# Baby Dragon Hatchling (BDH) - Complete Manual

**A Comprehensive Guide to Understanding, Using, and Building with BDH**

Version 1.0 | January 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Getting Started with Google Colab](#2-getting-started-with-google-colab)
3. [Understanding BDH Architecture](#3-understanding-bdh-architecture)
4. [Basic Usage Guide](#4-basic-usage-guide)
5. [Training Your First Model](#5-training-your-first-model)
6. [Text Generation](#6-text-generation)
7. [Experimentation Guide](#7-experimentation-guide)
8. [Advanced Topics](#8-advanced-topics)
9. [Building Applications with BDH](#9-building-applications-with-bdh)
10. [Growing Your Models](#10-growing-your-models)
11. [Troubleshooting](#11-troubleshooting)
12. [Resources & References](#12-resources--references)

---

## 1. Introduction

### What is Baby Dragon Hatchling?

Baby Dragon Hatchling (BDH) is a revolutionary neural network architecture that bridges the gap between:
- **Deep Learning** (transformers, GPT-like models)
- **Neuroscience** (biological brain computation)

Unlike traditional "black box" AI models, BDH is:
- ✅ **Interpretable**: You can understand what neurons are doing
- ✅ **Biologically inspired**: Uses principles from real brain networks
- ✅ **Performant**: Matches GPT-2 scale transformers (10M-1B parameters)
- ✅ **Sparse**: Activations are sparse and positive, like real neurons

### Key Innovation

Traditional transformers use **attention mechanisms imposed from above**.

BDH's attention **emerges naturally** from neuron-level interactions, similar to how attention works in biological brains.

### Research Foundation

- **Paper**: [The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain](https://arxiv.org/abs/2509.26507)
- **Authors**: A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz (Pathway)
- **Year**: 2025

### Why "Baby Dragon Hatchling"?

The model represents an early stage in understanding how intelligence emerges from simple neuron interactions - like a dragon hatching from an egg, showing the first signs of its eventual power.

---

## 2. Getting Started with Google Colab

### What is Google Colab?

Google Colab (Colaboratory) is a **free cloud-based Jupyter notebook environment** that provides:
- Free GPU access (NVIDIA Tesla T4 or similar)
- Python environment with pre-installed libraries
- No setup required - runs in your browser
- Free storage and compute (with limitations)

### Is Google Colab Really Free?

**YES! Google Colab is 100% FREE to use.**

**IMPORTANT - Don't Get Confused:**

When you visit Google Colab, you might see pages offering "Colab Pro" or "Colab Pro+" subscriptions. **You DO NOT need these for BDH!**

#### Free Tier (What You Need)
- ✅ **Cost**: $0 - Completely free
- ✅ **Credit card**: Not required
- ✅ **GPU access**: Yes, free NVIDIA GPU (Tesla T4 or similar)
- ✅ **Good for**: Learning, experimenting, training BDH
- ⚠️ **Limitations**:
  - 12-hour maximum session length
  - 90-minute idle timeout
  - GPU availability not guaranteed during peak times
  - Temporary storage (files deleted when session ends)

#### Paid Tiers (Optional - Not Needed)
- **Colab Pro** (~$10/month): Faster GPUs, longer sessions, priority access
- **Colab Pro+** (~$50/month): Even better resources
- **You don't need these** unless you're doing serious research or production work

#### How to Access the FREE Version

**Correct URL:**
```
https://colab.research.google.com
```

**What to do:**
1. Visit the URL above
2. Sign in with your Google account (Gmail)
3. Click "File" → "New notebook" or "Upload notebook"
4. Start using for FREE!

**What to avoid:**
- Don't click "Upgrade" or "Subscribe" buttons
- Don't go to `/signup` pages showing paid plans
- If you see pricing, just close it - you don't need to pay

The free tier is perfectly sufficient for:
- Learning BDH
- Training on toy datasets (Shakespeare, small text files)
- Experimenting with model architectures
- Generating text
- All exercises in this manual

### Step-by-Step Setup

#### Step 1: Access Google Colab

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account
3. You'll see the Colab welcome screen

#### Step 2: Upload the BDH Notebook

**Option A: Upload from your computer**
1. Download `BDH_Colab_Tutorial.ipynb` from this directory
2. In Colab, click **File → Upload notebook**
3. Select the downloaded `.ipynb` file
4. Click **Open**

**Option B: Import from GitHub**
1. In Colab, click **File → Open notebook**
2. Click the **GitHub** tab
3. Paste: `https://github.com/pathwaycom/bdh`
4. Select `BDH_Colab_Tutorial.ipynb` if available

**Option C: Start from scratch**
1. Click **File → New notebook**
2. Copy code from this manual section-by-section

#### Step 3: Enable GPU Acceleration

**CRITICAL STEP - Don't skip this!**

1. Click **Runtime → Change runtime type**
2. Under "Hardware accelerator", select **GPU** (default is None)
3. Click **Save**
4. Your notebook will restart with GPU enabled

To verify GPU is working, run:
```python
!nvidia-smi
```

You should see GPU information displayed.

#### Step 4: Understand the Interface

**Colab Interface Components:**

```
┌─────────────────────────────────────────────┐
│  File  Edit  View  Insert  Runtime  Tools   │ ← Menu bar
├─────────────────────────────────────────────┤
│  + Code    + Text                           │ ← Add cells
├─────────────────────────────────────────────┤
│  [ ] Code cell                              │
│      import torch                           │
│      print("Hello")                         │ ← Click ▶ to run
│                                             │
├─────────────────────────────────────────────┤
│  Output: Hello                              │ ← Output appears here
└─────────────────────────────────────────────┘
```

**Key shortcuts:**
- `Shift + Enter`: Run current cell and move to next
- `Ctrl + Enter`: Run current cell and stay
- `Ctrl + M B`: Insert cell below
- `Ctrl + M A`: Insert cell above

#### Step 5: Session Management

**Important Colab Limitations:**

- **Session timeout**: 12 hours maximum
- **Idle timeout**: ~90 minutes of inactivity
- **GPU availability**: Not guaranteed during peak times
- **Storage**: Temporary (deleted when session ends)

**Best Practices:**
- Save your work frequently
- Download important files before session ends
- Don't rely on Colab for permanent storage

---

## 3. Understanding BDH Architecture

### High-Level Concept

BDH is built on these biological principles:

#### 1. Scale-Free Network Topology

Real brains don't connect every neuron to every other neuron. They use **scale-free networks**:

```
Traditional Neural Net:      BDH Network:
    O─O─O─O                    O    O
    │ │ │ │                   ╱│╲  ╱│
    O─O─O─O                  O O O O
    │ │ │ │                   ╲│╱  ╲│
    O─O─O─O                    O    O

    Dense, regular             Sparse, hub-based
```

**Benefits:**
- More efficient
- Mimics biological connectivity
- Enables emergent behavior

#### 2. Locally Interacting Neurons

Each "neuron particle" in BDH:
- Interacts only with nearby neurons
- Has excitatory or inhibitory properties
- Maintains local state

**Biological parallel:** Like neurons in your brain's cortex

#### 3. Hebbian Learning

"Neurons that fire together, wire together"

BDH implements **synaptic plasticity**:
- Connections strengthen with use
- Working memory emerges naturally
- Monosemantic (one neuron = one concept)

#### 4. Sparse Activations

Unlike transformers where all weights activate:
- BDH neurons fire sparsely
- Activations are positive (like real neurons)
- Easy to interpret which neurons are "active"

### Architecture Diagram

```
Input Text: "To be or not"
     ↓
[Token Embedding] (converts chars → vectors)
     ↓
[BDH Layer 1] ─── Excitatory neurons
     │            Inhibitory neurons
     │            Local interactions
     ↓
[BDH Layer 2] ─── Hebbian memory
     │            Attention emerges
     ↓
[BDH Layer N]
     ↓
[Output Head] (vectors → predictions)
     ↓
Prediction: " to be"
```

### Code Structure

The BDH implementation consists of:

**1. `bdh.py`** - Core model
```python
class BDHConfig:
    # Configuration for model architecture
    vocab_size: int = 256      # Number of tokens (byte-level)
    n_embd: int = 384         # Embedding dimension
    n_layer: int = 6          # Number of layers
    dropout: float = 0.2      # Dropout rate

class BDH(nn.Module):
    # Main model class
    def forward(x, targets=None):
        # Forward pass through BDH network

    def generate(idx, max_new_tokens):
        # Text generation
```

**2. `train.py`** - Training script
- Downloads Shakespeare dataset
- Configures training loop
- Saves checkpoints
- Generates sample text

### Key Differences from Transformers

| Feature | Transformer | BDH |
|---------|------------|-----|
| **Attention** | Explicit multi-head attention | Emerges from neuron interactions |
| **Connectivity** | Fully connected | Scale-free graph |
| **Activations** | Dense (all weights active) | Sparse (selected neurons) |
| **Interpretability** | Black box | Transparent neuron states |
| **Biological** | Not biologically inspired | Mimics brain principles |
| **Performance** | SOTA on many tasks | Matches GPT-2 scale |

---

## 4. Basic Usage Guide

### Running Your First BDH Model

#### In Google Colab:

**Step 1: Clone Repository**
```python
!git clone https://github.com/pathwaycom/bdh.git
%cd bdh
```

**Step 2: Install Dependencies**
```python
!pip install -r requirements.txt -q
```

**Step 3: Check Setup**
```python
import bdh
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Verify BDH loads
config = bdh.BDHConfig()
model = bdh.BDH(config)
print("BDH loaded successfully!")
```

**Step 4: Run Quick Training**
```python
!python train.py
```

This will:
1. Download tiny Shakespeare dataset (~1MB)
2. Train for 3000 iterations (~10-15 minutes on GPU)
3. Print loss every 100 steps
4. Generate sample text: "To be or..."

### Understanding Output

**Training output looks like:**
```
Using device: cuda with dtype bfloat16
Step: 0/3000 loss 4.62
Step: 100/3000 loss 2.84
Step: 200/3000 loss 2.23
...
Step: 2900/3000 loss 1.45
Training done, now generating a sample
To be or not to be the king,
And the world is the prince of the world...
```

**What this means:**
- **Loss decreasing**: Model is learning!
- **Starting ~4.5**: Random guessing
- **Ending ~1.4-1.5**: Good performance for character-level modeling
- **Generated text**: Not perfect, but shows language structure

---

## 5. Training Your First Model

### Training Configuration

Edit these parameters in `train.py` or notebook:

```python
# Model architecture
BDH_CONFIG = bdh.BDHConfig(
    vocab_size=256,    # Byte-level tokens (0-255)
    n_embd=384,        # Hidden dimension (larger = more capacity)
    n_layer=6,         # Depth (more layers = more abstraction)
    dropout=0.2        # Regularization (0.0-0.5)
)

# Training hyperparameters
BLOCK_SIZE = 512       # Context length (characters)
BATCH_SIZE = 32        # Samples per batch
MAX_ITERS = 3000       # Training iterations
LEARNING_RATE = 1e-3   # Step size (1e-4 to 1e-2)
WEIGHT_DECAY = 0.1     # L2 regularization
```

### What Each Parameter Does

#### Model Parameters

**`vocab_size`** (default: 256)
- Number of different tokens
- 256 = byte-level (all ASCII + extended)
- Smaller = faster, larger = more expressiveness

**`n_embd`** (default: 384)
- Dimension of neuron representations
- Larger = more learning capacity, slower training
- Try: 256 (small), 384 (default), 768 (large)

**`n_layer`** (default: 6)
- Number of BDH layers
- More layers = deeper abstraction
- Try: 4 (fast), 6 (default), 12 (large)

**`dropout`** (default: 0.2)
- Probability of dropping neurons during training
- Prevents overfitting
- 0.0 = no dropout, 0.5 = heavy dropout

#### Training Parameters

**`BLOCK_SIZE`** (default: 512)
- Context window (how many characters model sees)
- Larger = more context, more memory
- Must be ≤ data sequence length

**`BATCH_SIZE`** (default: 32)
- Training samples per iteration
- Larger = stable gradients, more memory
- Adjust based on GPU memory

**`MAX_ITERS`** (default: 3000)
- Total training steps
- More = better learning (up to a point)
- ~3000 good for toy datasets

**`LEARNING_RATE`** (default: 1e-3)
- How fast model updates weights
- Too high = unstable, too low = slow
- Use 1e-4 for large models, 1e-3 for small

### Training Workflow

#### Standard Training Process

**1. Prepare Data**
```python
import requests
import os

# Download dataset
data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"):
    with open("input.txt", "w") as f:
        f.write(requests.get(data_url).text)
```

**2. Initialize Model**
```python
import bdh
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = bdh.BDHConfig()
model = bdh.BDH(config).to(device)
model = torch.compile(model)  # Optimize for speed

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model has {total_params:,} parameters")
```

**3. Setup Optimizer**
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,           # Learning rate
    weight_decay=0.1    # L2 regularization
)
```

**4. Training Loop**
```python
import numpy as np

# Data loading
def get_batch(split='train'):
    data = np.memmap('input.txt', dtype=np.uint8, mode='r')
    if split == 'train':
        data = data[:int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)):]

    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# Training
model.train()
for step in range(MAX_ITERS):
    # Get batch
    x, y = get_batch('train')

    # Forward pass
    logits, loss = model(x, y)

    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Log progress
    if step % 100 == 0:
        print(f"Step {step}: loss {loss.item():.4f}")
```

**5. Save Model**
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
}, 'bdh_checkpoint.pt')
```

### Monitoring Training

**Good signs:**
- Loss steadily decreasing
- Generated text improves over time
- No NaN or Inf values

**Warning signs:**
- Loss increasing: Learning rate too high
- Loss not decreasing: Learning rate too low or model too small
- Loss goes to NaN: Numerical instability (reduce learning rate)

**Expected loss values:**
- Start: ~4.5 (random guessing)
- After 1000 steps: ~2.5-3.0
- After 3000 steps: ~1.4-1.8
- Well-trained: ~1.0-1.3

---

## 6. Text Generation

### Basic Generation

After training, generate text:

```python
model.eval()  # Switch to evaluation mode

# Create prompt
prompt = "To be or not to be"
prompt_tensor = torch.tensor(
    bytearray(prompt, "utf-8"),
    dtype=torch.long,
    device=device
).unsqueeze(0)

# Generate
with torch.no_grad():
    output = model.generate(
        prompt_tensor,
        max_new_tokens=200,  # Generate 200 characters
        top_k=10             # Sample from top 10 predictions
    )

# Decode
result = bytes(output.to(torch.uint8).cpu().squeeze(0)).decode(
    errors='backslashreplace'
)
print(result)
```

### Generation Parameters

#### `max_new_tokens`
- How many new characters to generate
- More = longer text
- Typical: 100-500

#### `top_k`
- Sample from top K most likely next tokens
- Lower = more conservative/focused
- Higher = more random/creative
- Recommended values:
  - `top_k=1`: Greedy (always pick best)
  - `top_k=5`: Focused generation
  - `top_k=10`: Balanced
  - `top_k=50`: Creative
  - `top_k=200`: Very random

### Advanced Generation Techniques

#### 1. Temperature Sampling

Modify the generation code to add temperature:

```python
def generate_with_temperature(model, prompt, max_tokens=200, temperature=1.0, top_k=10):
    prompt_tensor = torch.tensor(
        bytearray(prompt, "utf-8"),
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        idx = prompt_tensor
        for _ in range(max_tokens):
            # Get logits
            logits, _ = model(idx[:, -BLOCK_SIZE:])
            logits = logits[:, -1, :] / temperature  # Apply temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

    return bytes(idx.to(torch.uint8).cpu().squeeze(0)).decode(errors='backslashreplace')

# Usage:
# Low temperature = focused, high = creative
print(generate_with_temperature(model, "To be", temperature=0.5, top_k=10))  # Conservative
print(generate_with_temperature(model, "To be", temperature=1.0, top_k=10))  # Balanced
print(generate_with_temperature(model, "To be", temperature=2.0, top_k=10))  # Wild
```

**Temperature guide:**
- `0.1-0.5`: Very focused, repetitive
- `0.7-1.0`: Balanced, coherent
- `1.5-2.0`: Creative, sometimes nonsensical
- `>2.0`: Chaotic, random

#### 2. Prompt Engineering

**Good prompts:**
```python
prompts = [
    "ROMEO:",                    # Character dialogue
    "Act I, Scene 1",           # Play structure
    "To be or not to be,",      # Famous quotes
    "Once upon a time",         # Story beginning
    "First Citizen:",           # Crowd dialogue
]
```

**Tips:**
- Use capitalization matching training data
- Include punctuation
- Match the style you want

#### 3. Batch Generation

Generate multiple completions in parallel:

```python
def generate_multiple(model, prompt, n_samples=5, max_tokens=100, top_k=10):
    results = []
    for i in range(n_samples):
        result = generate_with_temperature(
            model, prompt,
            max_tokens=max_tokens,
            top_k=top_k,
            temperature=1.0
        )
        results.append(result)
        print(f"\n--- Sample {i+1} ---")
        print(result)
    return results

# Generate 5 different completions
samples = generate_multiple(model, "To be or not", n_samples=5)
```

---

## 7. Experimentation Guide

### Experiment 1: Model Size Comparison

Compare small vs large models:

```python
# Small model
small_config = bdh.BDHConfig(n_embd=256, n_layer=4)
small_model = bdh.BDH(small_config).to(device)

# Large model
large_config = bdh.BDHConfig(n_embd=768, n_layer=12)
large_model = bdh.BDH(large_config).to(device)

# Count parameters
small_params = sum(p.numel() for p in small_model.parameters())
large_params = sum(p.numel() for p in large_model.parameters())

print(f"Small model: {small_params:,} parameters")
print(f"Large model: {large_params:,} parameters")
print(f"Ratio: {large_params/small_params:.1f}x")

# Train both and compare loss curves
```

### Experiment 2: Learning Rate Search

Find optimal learning rate:

```python
learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
results = {}

for lr in learning_rates:
    print(f"\nTesting LR: {lr}")

    model = bdh.BDH(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    for step in range(500):  # Quick test
        x, y = get_batch('train')
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            losses.append(loss.item())

    results[lr] = losses
    print(f"Final loss: {losses[-1]:.4f}")

# Plot results
import matplotlib.pyplot as plt
for lr, losses in results.items():
    plt.plot(losses, label=f'LR={lr}')
plt.legend()
plt.xlabel('Step (x100)')
plt.ylabel('Loss')
plt.title('Learning Rate Comparison')
plt.show()
```

### Experiment 3: Different Datasets

Train on different text types:

**Code dataset:**
```python
# Download Python code
code_url = "https://raw.githubusercontent.com/python/cpython/main/Lib/os.py"
with open("input.txt", "w") as f:
    f.write(requests.get(code_url).text)

# Train BDH on code
# ... train as normal ...

# Generate code
generate_text("def fibonacci(")
```

**Poetry dataset:**
```python
# Use Project Gutenberg poetry
poetry_url = "https://www.gutenberg.org/files/1322/1322-0.txt"  # Example
# ... download and train ...
```

**Your own data:**
```python
# Create custom dataset
with open("input.txt", "w") as f:
    f.write("""
    Your custom text here.
    Can be anything: stories, documentation, etc.
    The model will learn your writing style.
    """)
```

### Experiment 4: Architecture Ablations

Test what makes BDH work:

```python
# Vary number of layers
for n_layers in [2, 4, 6, 8, 12]:
    config = bdh.BDHConfig(n_layer=n_layers)
    # ... train and measure ...

# Vary embedding dimension
for n_embd in [128, 256, 384, 512, 768]:
    config = bdh.BDHConfig(n_embd=n_embd)
    # ... train and measure ...

# Vary dropout
for dropout in [0.0, 0.1, 0.2, 0.3, 0.5]:
    config = bdh.BDHConfig(dropout=dropout)
    # ... train and measure ...
```

### Experiment 5: Interpretability Analysis

Analyze what neurons learn:

```python
# Hook to capture activations
activations = {}

def hook_fn(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks
model.transformer.h[0].register_forward_hook(hook_fn('layer_0'))
model.transformer.h[1].register_forward_hook(hook_fn('layer_1'))

# Run inference
with torch.no_grad():
    model.eval()
    prompt = torch.tensor(bytearray("Hello", "utf-8"), dtype=torch.long, device=device).unsqueeze(0)
    _ = model(prompt)

# Analyze activations
for name, act in activations.items():
    print(f"{name}: shape={act.shape}, sparsity={(act == 0).float().mean():.2%}")

    # Visualize
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 3))
    plt.imshow(act[0].cpu().numpy(), aspect='auto', cmap='viridis')
    plt.title(f'Activations: {name}')
    plt.colorbar()
    plt.show()
```

---

## 8. Advanced Topics

### Transfer Learning

Fine-tune a pre-trained model on new data:

```python
# Load checkpoint
checkpoint = torch.load('bdh_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Freeze some layers
for param in model.transformer.h[:3].parameters():  # Freeze first 3 layers
    param.requires_grad = False

# Fine-tune on new data
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4  # Lower learning rate for fine-tuning
)

# Train as normal
```

### Multi-GPU Training

Scale to multiple GPUs:

```python
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
```

### Gradient Accumulation

Train larger effective batch sizes:

```python
GRADIENT_ACCUMULATION_STEPS = 4

optimizer.zero_grad()
for step in range(MAX_ITERS):
    x, y = get_batch('train')
    logits, loss = model(x, y)
    loss = loss / GRADIENT_ACCUMULATION_STEPS  # Scale loss
    loss.backward()

    if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Learning Rate Scheduling

Improve training with learning rate decay:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=MAX_ITERS)

for step in range(MAX_ITERS):
    # ... training step ...
    scheduler.step()

    if step % 100 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Step {step}: LR={current_lr:.6f}")
```

### Evaluation Metrics

Measure model quality:

```python
import math

def calculate_perplexity(model, data_loader):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for x, y in data_loader:
            logits, loss = model(x, y)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

# Lower perplexity = better model
ppl = calculate_perplexity(model, test_loader)
print(f"Perplexity: {ppl:.2f}")
```

### Custom Loss Functions

Experiment with different objectives:

```python
class BDHWithCustomLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bdh = bdh.BDH(config)

    def forward(self, x, y):
        logits, ce_loss = self.bdh(x, y)

        # Add custom regularization
        l1_loss = sum(p.abs().sum() for p in self.parameters())

        # Combined loss
        total_loss = ce_loss + 0.001 * l1_loss

        return logits, total_loss
```

---

## 9. Building Applications with BDH

### Application 1: Writing Assistant

```python
class WritingAssistant:
    def __init__(self, model_path):
        checkpoint = torch.load(model_path)
        self.config = checkpoint['config']
        self.model = bdh.BDH(self.config).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def complete_text(self, prompt, num_suggestions=3, length=100):
        \"\"\"Generate multiple completions for writing.\"\"\"
        suggestions = []

        for i in range(num_suggestions):
            completion = self.generate(prompt, max_tokens=length, top_k=10)
            suggestions.append(completion)

        return suggestions

    def generate(self, prompt, max_tokens=100, top_k=10):
        prompt_tensor = torch.tensor(
            bytearray(prompt, "utf-8"),
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        with torch.no_grad():
            output = self.model.generate(prompt_tensor, max_new_tokens=max_tokens, top_k=top_k)

        return bytes(output.to(torch.uint8).cpu().squeeze(0)).decode(errors='backslashreplace')

# Usage
assistant = WritingAssistant('bdh_checkpoint.pt')
suggestions = assistant.complete_text("Once upon a time", num_suggestions=3)

for i, suggestion in enumerate(suggestions, 1):
    print(f"\n--- Suggestion {i} ---")
    print(suggestion)
```

### Application 2: Code Completion

```python
class CodeCompleter:
    def __init__(self, model_path):
        # Load model trained on code
        checkpoint = torch.load(model_path)
        self.model = bdh.BDH(checkpoint['config']).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def complete_code(self, code_snippet, max_length=200):
        \"\"\"Complete a code snippet.\"\"\"
        # Use greedy decoding for code (top_k=1)
        prompt_tensor = torch.tensor(
            bytearray(code_snippet, "utf-8"),
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        with torch.no_grad():
            output = self.model.generate(prompt_tensor, max_new_tokens=max_length, top_k=1)

        return bytes(output.to(torch.uint8).cpu().squeeze(0)).decode(errors='backslashreplace')

# Usage
completer = CodeCompleter('bdh_code_checkpoint.pt')
completion = completer.complete_code("def fibonacci(n):\\n    if n <= 1:")
print(completion)
```

### Application 3: Style Transfer

```python
def train_style_model(style_text_file):
    \"\"\"Train BDH on specific writing style.\"\"\"
    # Load your style examples
    with open(style_text_file, 'r') as f:
        style_data = f.read()

    # Save as input.txt
    with open('input.txt', 'w') as f:
        f.write(style_data)

    # Train model
    config = bdh.BDHConfig()
    model = bdh.BDH(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # ... training loop ...

    return model

# Train on Shakespeare
shakespeare_model = train_style_model('shakespeare.txt')

# Train on technical docs
tech_model = train_style_model('technical_docs.txt')

# Generate in different styles
print("Shakespeare style:", generate_text(shakespeare_model, "The king"))
print("Technical style:", generate_text(tech_model, "The algorithm"))
```

### Application 4: Interactive Chatbot

```python
class SimpleChatbot:
    def __init__(self, model_path):
        checkpoint = torch.load(model_path)
        self.model = bdh.BDH(checkpoint['config']).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.conversation_history = ""

    def chat(self, user_input, max_response_length=100):
        \"\"\"Simple turn-based chat.\"\"\"
        # Add user input to history
        self.conversation_history += f"User: {user_input}\\n"

        # Generate response
        prompt = self.conversation_history + "Bot: "
        response = self.generate_from_history(prompt, max_response_length)

        # Extract just the bot response
        bot_response = response.split("\\n")[0]

        # Add to history
        self.conversation_history += f"Bot: {bot_response}\\n"

        return bot_response

    def generate_from_history(self, prompt, max_tokens):
        prompt_tensor = torch.tensor(
            bytearray(prompt, "utf-8"),
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        with torch.no_grad():
            output = self.model.generate(prompt_tensor, max_new_tokens=max_tokens, top_k=10)

        return bytes(output.to(torch.uint8).cpu().squeeze(0)).decode(errors='backslashreplace')

# Usage
bot = SimpleChatbot('bdh_dialog_checkpoint.pt')

while True:
    user_msg = input("You: ")
    if user_msg.lower() in ['quit', 'exit', 'bye']:
        break

    response = bot.chat(user_msg)
    print(f"Bot: {response}")
```

---

## 10. Growing Your Models

### Strategy 1: Incremental Scaling

Start small, grow gradually:

```python
# Phase 1: Tiny model for quick iteration
tiny_config = bdh.BDHConfig(n_embd=128, n_layer=4)
tiny_model = train_model(tiny_config, max_iters=1000)

# Phase 2: Small model
small_config = bdh.BDHConfig(n_embd=256, n_layer=6)
small_model = train_model(small_config, max_iters=3000)

# Phase 3: Medium model
medium_config = bdh.BDHConfig(n_embd=384, n_layer=8)
medium_model = train_model(medium_config, max_iters=5000)

# Phase 4: Large model
large_config = bdh.BDHConfig(n_embd=768, n_layer=12)
large_model = train_model(large_config, max_iters=10000)
```

### Strategy 2: Curriculum Learning

Train on progressively harder data:

```python
# Stage 1: Simple, repetitive text
with open('input.txt', 'w') as f:
    f.write("hello " * 10000)
train_model(model, max_iters=500)

# Stage 2: Simple sentences
with open('input.txt', 'w') as f:
    f.write("The cat sat. The dog ran. " * 1000)
train_model(model, max_iters=1000)

# Stage 3: Complex text (Shakespeare)
# ... load complex dataset ...
train_model(model, max_iters=5000)
```

### Strategy 3: Knowledge Distillation

Train small model from large model:

```python
# Train large teacher model
teacher_config = bdh.BDHConfig(n_embd=768, n_layer=12)
teacher = bdh.BDH(teacher_config).to(device)
# ... train teacher ...

# Train small student model
student_config = bdh.BDHConfig(n_embd=256, n_layer=4)
student = bdh.BDH(student_config).to(device)

# Distillation training
teacher.eval()
student.train()

for step in range(MAX_ITERS):
    x, y = get_batch('train')

    # Get teacher predictions
    with torch.no_grad():
        teacher_logits, _ = teacher(x)

    # Student predictions
    student_logits, _ = student(x)

    # Distillation loss (match teacher's distribution)
    distill_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    )

    distill_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Strategy 4: Continual Learning

Keep learning from new data without forgetting:

```python
# Initial training
model = train_on_dataset('shakespeare.txt', max_iters=5000)

# Add new knowledge
# Use lower learning rate to preserve old knowledge
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # Lower LR
model = train_on_dataset('modern_text.txt', max_iters=2000)

# Add more knowledge
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # Even lower
model = train_on_dataset('technical_docs.txt', max_iters=1000)
```

### Scaling Guidelines

| Model Size | n_embd | n_layer | Parameters | GPU Memory | Training Time (3k iters) |
|------------|--------|---------|------------|------------|-------------------------|
| Tiny       | 128    | 4       | ~1M        | ~1 GB      | ~5 min                  |
| Small      | 256    | 6       | ~5M        | ~2 GB      | ~10 min                 |
| Medium     | 384    | 8       | ~15M       | ~4 GB      | ~20 min                 |
| Large      | 768    | 12      | ~60M       | ~8 GB      | ~45 min                 |
| XL         | 1024   | 16      | ~150M      | ~16 GB     | ~90 min                 |

**Hardware recommendations:**
- Tiny/Small: Free Colab
- Medium: Colab Pro or local GPU (RTX 3060+)
- Large: Colab Pro+ or high-end GPU (RTX 3090+)
- XL: Multi-GPU or cloud (A100)

---

## 11. Troubleshooting

### Common Issues

#### Issue 1: "RuntimeError: CUDA out of memory"

**Cause:** Model or batch size too large for GPU

**Solutions:**
```python
# Reduce batch size
BATCH_SIZE = 16  # or 8, 4

# Reduce model size
config = bdh.BDHConfig(n_embd=256, n_layer=4)

# Enable gradient checkpointing (if implemented)
model.gradient_checkpointing_enable()

# Use CPU (slow)
device = torch.device('cpu')
```

#### Issue 2: Loss is NaN

**Cause:** Numerical instability

**Solutions:**
```python
# Reduce learning rate
LEARNING_RATE = 1e-4

# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Check for inf/nan in data
assert not torch.isnan(x).any()
assert not torch.isinf(x).any()
```

#### Issue 3: Loss not decreasing

**Cause:** Learning rate too low, model too small, or data issue

**Solutions:**
```python
# Increase learning rate
LEARNING_RATE = 3e-3

# Increase model size
config = bdh.BDHConfig(n_embd=512, n_layer=8)

# Train longer
MAX_ITERS = 10000

# Check data is loading correctly
x, y = get_batch('train')
print(x.shape, y.shape)
print(x[0][:50])  # Print first sample
```

#### Issue 4: Model generates garbage

**Cause:** Undertrained or wrong generation parameters

**Solutions:**
```python
# Train longer
MAX_ITERS = 5000

# Reduce top_k for more focused generation
output = model.generate(prompt, top_k=3)

# Use lower temperature
generate_with_temperature(model, prompt, temperature=0.7)

# Check training loss is low (<1.5)
```

#### Issue 5: Colab session disconnects

**Cause:** Idle timeout or 12-hour limit

**Solutions:**
```python
# Save checkpoints frequently
if step % 500 == 0:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }, f'checkpoint_step_{step}.pt')

# Resume from checkpoint
checkpoint = torch.load('checkpoint_step_2500.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_step = checkpoint['step']

for step in range(start_step, MAX_ITERS):
    # ... continue training ...
```

#### Issue 6: Slow training on CPU

**Cause:** No GPU available

**Solutions:**
- Use Google Colab with GPU (Runtime → Change runtime type → GPU)
- Reduce model size
- Reduce MAX_ITERS
- Consider cloud GPU services (Colab Pro, AWS, etc.)

### Getting Help

**Resources:**
- GitHub Issues: https://github.com/pathwaycom/bdh/issues
- Paper: https://arxiv.org/abs/2509.26507
- Video: https://www.youtube.com/watch?v=mfV44-mtg7c

**Before asking for help:**
1. Check this manual's troubleshooting section
2. Verify GPU is enabled (in Colab)
3. Try reducing model size
4. Check data is loading correctly
5. Save and share your error message

---

## 12. Resources & References

### Official Resources

**Repository:**
- GitHub: https://github.com/pathwaycom/bdh
- License: See LICENSE.md

**Research Paper:**
- Title: "The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain"
- arXiv: https://arxiv.org/abs/2509.26507
- Authors: A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz
- Organization: Pathway (https://pathway.com)

**Media Coverage:**
- Forbes: [Can AI Learn and Evolve Like a Brain?](https://www.forbes.com/sites/victordey/2025/10/08/can-ai-learn-and-evolve-like-a-brain-pathways-bold-research-thinks-so/)
- Semafor: [New AI Research Claims to Be Getting Closer to Modeling Human Brain](https://www.semafor.com/article/10/01/2025/new-ai-research-claims-to-be-getting-closer-to-modeling-human-brain)
- SuperDataScience Podcast: [YouTube](https://www.youtube.com/watch?v=mfV44-mtg7c)

**Community Ports:**
- Burn (Rust): https://github.com/mosure/burn_dragon_hatchling
- MLX (Apple Silicon): https://github.com/severian42/bdh
- Other implementations: See main README

### Learning Resources

**Machine Learning Fundamentals:**
- Deep Learning Book: https://www.deeplearningbook.org
- Neural Networks (3Blue1Brown): https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- Stanford CS231n: https://cs231n.github.io

**Transformers & Language Models:**
- Attention Is All You Need (paper): https://arxiv.org/abs/1706.03762
- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- Andrej Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT

**Neuroscience:**
- Scale-free networks in the brain: Multiple papers on arXiv
- Hebbian learning: "The Organization of Behavior" by Donald Hebb
- Sparse coding: Research by Bruno Olshausen

### Tools & Libraries

**Python/PyTorch:**
- PyTorch: https://pytorch.org
- PyTorch tutorials: https://pytorch.org/tutorials
- NumPy: https://numpy.org

**Google Colab:**
- Colab homepage: https://colab.research.google.com
- Colab tips: https://colab.research.google.com/notebooks/basic_features_overview.ipynb
- Colab FAQ: https://research.google.com/colaboratory/faq.html

**Visualization:**
- Matplotlib: https://matplotlib.org
- TensorBoard: https://www.tensorflow.org/tensorboard
- Weights & Biases: https://wandb.ai

### Datasets

**Text Datasets:**
- Tiny Shakespeare: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- Project Gutenberg: https://www.gutenberg.org
- Wikipedia dumps: https://dumps.wikimedia.org
- OpenWebText: https://openwebtext2.readthedocs.io

**Code Datasets:**
- GitHub code: https://github.com
- The Stack: https://huggingface.co/datasets/bigcode/the-stack

### Related Projects

**Similar Architectures:**
- nanoGPT: https://github.com/karpathy/nanoGPT
- minGPT: https://github.com/karpathy/minGPT
- GPT-2: https://github.com/openai/gpt-2

**Neuroscience-inspired AI:**
- Sparse Distributed Representations: https://numenta.com
- Capsule Networks: https://arxiv.org/abs/1710.09829

### Citation

If you use BDH in research, please cite:

```bibtex
@article{kosowski2025dragon,
  title={The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain},
  author={Kosowski, Adrian and Uzna{\'n}ski, Przemys{\l}aw and Chorowski, Jan and Stamirowska, Zofia and Bartoszkiewicz, Micha{\l}},
  journal={arXiv preprint arXiv:2509.26507},
  year={2025}
}
```

---

## Appendix: Quick Reference

### Key Commands

```python
# Setup
!git clone https://github.com/pathwaycom/bdh.git
%cd bdh
!pip install -r requirements.txt

# Train
!python train.py

# Custom training
config = bdh.BDHConfig(n_embd=384, n_layer=6)
model = bdh.BDH(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Generate
prompt = torch.tensor(bytearray("Hello", "utf-8"), dtype=torch.long, device=device).unsqueeze(0)
output = model.generate(prompt, max_new_tokens=200, top_k=10)
result = bytes(output.to(torch.uint8).cpu().squeeze(0)).decode()

# Save
torch.save({'model_state_dict': model.state_dict()}, 'model.pt')

# Load
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Configuration Cheat Sheet

```python
# Tiny (fast experiments)
bdh.BDHConfig(n_embd=128, n_layer=4)

# Small (default)
bdh.BDHConfig(n_embd=384, n_layer=6)

# Medium (better quality)
bdh.BDHConfig(n_embd=512, n_layer=8)

# Large (high quality, needs good GPU)
bdh.BDHConfig(n_embd=768, n_layer=12)
```

### Training Tips

- ✅ Start with default config
- ✅ Use GPU (Colab: Runtime → Change runtime type → GPU)
- ✅ Monitor loss (should decrease)
- ✅ Save checkpoints frequently
- ✅ Lower LR if loss goes to NaN
- ✅ Increase iterations for better quality

### Generation Tips

- ✅ Use `top_k=1` for deterministic output
- ✅ Use `top_k=10` for balanced creativity
- ✅ Use `temperature<1.0` for focused output
- ✅ Use `temperature>1.0` for creative output
- ✅ Test multiple prompts
- ✅ Generate multiple samples and pick best

---

**End of Manual**

For the latest updates, visit: https://github.com/pathwaycom/bdh

**Happy Dragon Hatching! 🐉**
