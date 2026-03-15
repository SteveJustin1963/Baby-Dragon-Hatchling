# Quick Start Guide - BDH on Google Colab

**Get started with Baby Dragon Hatchling in 5 minutes!**

---

## Is This Really Free?

**YES! 100% FREE. NO CREDIT CARD REQUIRED.**

Google Colab's free tier gives you:
- ✅ Free GPU access
- ✅ Free compute time
- ✅ No payment needed
- ✅ Just need a Google account

**Don't get confused:** If you see "Colab Pro" or pricing pages, just ignore them. You don't need to pay anything for BDH!

**Correct URL to use (FREE):**
```
https://colab.research.google.com
```

---

## What You Need

1. **Google Account** (Gmail) - Free to create at gmail.com
2. **Web Browser** (Chrome, Firefox, Safari, etc.)
3. **The notebook file**: `BDH_Colab_Tutorial.ipynb` (in this directory)
4. **5-30 minutes** of your time

---

## Step-by-Step Instructions

### Step 1: Go to Google Colab

Open your browser and visit:
```
https://colab.research.google.com
```

Sign in with your Google account.

### Step 2: Upload the Notebook

1. Click **"File"** → **"Upload notebook"**
2. Click **"Choose File"**
3. Select **`BDH_Colab_Tutorial.ipynb`** from this directory
4. Click **"Open"**

The notebook will open in Colab!

### Step 3: Enable GPU (CRITICAL!)

**Don't skip this step or training will be very slow!**

1. Click **"Runtime"** (in the menu bar)
2. Click **"Change runtime type"**
3. Under **"Hardware accelerator"**, select **"GPU"**
4. Click **"Save"**

### Step 4: Run the Notebook

Now just click through the cells from top to bottom:

1. Click the **▶ Play button** on the first cell (or press `Shift+Enter`)
2. Wait for it to finish
3. Click the **▶ Play button** on the next cell
4. Repeat!

**The notebook will:**
- Install dependencies (~2-3 minutes)
- Train BDH on Shakespeare (~15-20 minutes on GPU)
- Generate text samples
- Show you how to experiment

### Step 5: Experiment!

Once training is done, you can:
- Try different prompts in the "Interactive Generation" cell
- Train longer for better results
- Try different model sizes
- Generate creative text

---

## Need Help?

### Can't upload the notebook?

**Alternative method:**
1. Open Google Colab
2. Click **"File"** → **"New notebook"**
3. Open `BDH_Colab_Tutorial.ipynb` in a text editor
4. Copy and paste each cell manually

### GPU not available?

Check your settings:
- Runtime → Change runtime type → GPU should be selected
- If GPU is grayed out, you might need to wait (free tier has limits)
- Try again in a few hours

### Training is slow?

Make sure:
- GPU is enabled (see Step 3)
- Run `!nvidia-smi` in a cell - you should see GPU info
- If still slow, you might be on CPU

### Session disconnected?

This happens after ~90 minutes of idle time:
- Just reconnect and re-run cells
- Your work is saved in Colab
- For long training, save checkpoints (see MANUAL.md)

---

## What's Next?

After running the basic notebook:

1. **Read the full manual**: `MANUAL.md` has everything you need to know
2. **Experiment**: Try the experiments in the notebook
3. **Build something**: Use BDH for your own projects
4. **Learn more**: Read the research paper linked in the manual

---

## File Guide

In this directory you have:

| File | Purpose |
|------|---------|
| `BDH_Colab_Tutorial.ipynb` | Interactive notebook for Google Colab |
| `MANUAL.md` | Complete 12-chapter guide (100+ pages) |
| `QUICK_START.md` | This file - get started fast |
| `bdh.py` | The BDH model code |
| `train.py` | Training script (for local use) |
| `README.md` | Original project README |

---

## Tips for Success

✅ **Always enable GPU** in Colab (Runtime → Change runtime type → GPU)
✅ **Start with the default settings** in the notebook
✅ **Save your work** - Download important files before session ends
✅ **Be patient** - Training takes 15-30 minutes on GPU
✅ **Experiment** - Try different prompts and parameters
✅ **Read the manual** - It has everything: troubleshooting, advanced topics, applications

---

## Common Questions

**Q: Is this really free?**
A: Yes! Google Colab's free tier includes GPU access.

**Q: How long does training take?**
A: ~15-30 minutes on GPU, hours on CPU.

**Q: Do I need to know Python?**
A: Basic Python helps, but you can follow along by running cells.

**Q: Can I use my own text data?**
A: Yes! See MANUAL.md section 7 for how to use custom datasets.

**Q: Will my trained model be saved?**
A: Only if you download it before the session ends. See the "Save Model" cell.

**Q: I got an error, what do I do?**
A: Check MANUAL.md section 11 (Troubleshooting) for solutions.

---

**Ready? Let's go! 🐉**

Open Google Colab and upload `BDH_Colab_Tutorial.ipynb` to get started!
