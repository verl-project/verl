# 🧬 LLM-Guided De Novo Protein Design Environment

**De novo protein binder design** is one of the hardest problems in bioengineering: you're tasked with inventing an amino acid sequence that folds into a 3D structure that binds to a given target protein. This environment lets **Large Language Models (LLMs)** tackle that problem using reinforcement learning (RL) — not by predicting sequences blindly, but by *learning to use the right tools in the right order* to produce functioning binders.

---

## 🤖 Why LLM-based RL Instead of Classic RL?

Classic RL works well for Atari, but it could never work for de novo protein binder design. Why?

- **Simulation is slow.** Each step—AlphaFold, RFdiffusion, ProteinMPNN—can take minutes. You don’t get to run millions of episodes like in classic RL.
- **State/action spaces are vast and weird.** Proteins are not 2D boards or pixel arrays. Designing them involves sequences, structures, config files, hotspots, and domain hacks.
- **Heuristics and intuition matter.** LLMs are pretrained on a *world model*—language, code, protein sequences, scientific papers. They come in with baked-in priors that help them reason, even under sparse rewards.

**Classic RL policy networks?** They’d need to learn everything from scratch, which is impossible!

---

## 🧪 The Protein Design Pipeline

Each episode consists of an LLM navigating a 4-step design pipeline, using state-of-the-art tools as function calls:

### Step 1: Target Sequence → Structure (`AlphaFold`)
- **Input:** Target protein sequence
- **Output:** 3D `.pdb` file (structure)
- **Reward:** Format validity

### Step 2: Target Structure → Binder Backbone (`RFdiffusion`)
- **Input:** `.pdb` file of target
- **Output:** `.pdb` backbone of potential binder
- **Reward:** Format validity

### Step 3: Backbone → Full Binder Sequence (`ProteinMPNN`)
- **Input:** Binder backbone
- **Output:** `.fasta` with side chains
- **Reward:** Format validity

### Step 4: Evaluate Binding (`AlphaFold-Multimer`)
- **Input:** Target + binder sequences
- **Output:** Complex structure prediction
- **Reward:**
  - Format OK
  - No steric clashes
  - **Bonus:** Contact interface, binding affinity metrics (Not yet implemented)

---

## 🏆 Reward Function

The reward is cumulative:
- **+0.2**: Successfully generate output in correct format at each step
- **+0.0 to +1.0:** Structural reward based on complex validity smoothly interpolated on AlphaFold2 multimere confidence
- **+1**: High predicted binding affinity (Not yet implemented)

Sparse, but real. LLMs must *plan* tool use, not just spam actions.

---

## 🔧 Setup

Access to hosted NVIDIA APIs:
```env
NVIDIA_NIM_API_KEY="YOUR_API_KEY"
```
