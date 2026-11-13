# NSF-PINN Model

This code serves to recreate the PINN model that is used in the paper titled "NSF-PINN: Physics-informed source-filter decomposition with sharpness-aware minimization for generalizable audio deepfake detection"
doi: https://doi-org.ezproxy.bgsu.edu/10.1016/j.cviu.2026.104875

# NSF-PINN: Physics-Informed Neural Source-Filter for Audio Deepfake Detection

This repository contains the PyTorch implementation of **NSF-PINN** (Neural Source-Filter Physics-Informed Neural Network), an advanced audio deepfake detection system.

By integrating domain knowledge from classical speech production theory (the source-filter model) with modern self-supervised learning and Sharpness-Aware Minimization (SAM), NSF-PINN successfully generalizes to unseen spoofing conditions, codecs, and acoustic environments.

## 🧠 How the Model Works: Stage-by-Stage

Modern text-to-speech and voice conversion systems often generate audio that sounds perfect to the human ear but violates the physical constraints of the human vocal tract. NSF-PINN detects these microscopic violations through five distinct stages:

### Stage 1: The Self-Supervised Learning (SSL) Backbone

* **What it does:** The model takes a raw $16\text{ kHz}$ audio waveform and passes it through a pre-trained **WavLM-Base+** transformer.

* **Details:** We freeze the CNN feature extractor and the first 6 transformer layers to retain stable, generalized acoustic representations. The active layers output a sequence of contextual hidden states, denoted as $H \in \mathbb{R}^{L \times 768}$, where $L$ is the number of temporal frames. These features encode rich phonetic and prosodic information.

### Stage 2: The Source-Filter Decomposition Branch

* **What it does:** Human speech is physically produced by pushing air through the vocal cords (the *Source* / glottal excitation) and shaping it with the mouth and throat (the *Filter* / vocal tract resonance). This branch forces the model to mathematically separate these two components.

* **Details:** A shared temporal Multi-Layer Perceptron (MLP) processes the hidden states $H$. Two independent linear projection heads then split this representation into a frame-level source sequence $S \in \mathbb{R}^{L \times 128}$ and a filter sequence $F \in \mathbb{R}^{L \times 128}$.

### Stage 3: Physics-Informed Regularization (The Core Innovation)

* **What it does:** To ensure the separated $S$ and $F$ vectors represent *actual* physics (and not just arbitrary math), the network applies three auxiliary loss functions during training:

  1. **Reconstruction Consistency (**$\mathcal{L}_{\text{rec}}$**):** Ensures that the combined source and filter can perfectly reconstruct the original WavLM embedding, enforcing the invertibility of the source-filter model.

  2. **Temporal Smoothness (**$\mathcal{L}_{\text{smo}}$**):** Human articulators (lips, tongue) have physical mass and cannot teleport. This loss penalizes physically impossible, high-frequency frame-to-frame jumps in the filter and source trajectories.

  3. **Source-Filter Orthogonality (**$\mathcal{L}_{\text{orth}}$**):** In natural speech, glottal excitation and vocal tract resonance are biologically independent. This loss minimizes the cosine similarity between $S$ and $F$. Deepfakes often heavily entangle these features, triggering this penalty.

### Stage 4: Feature Fusion and Classification

* **What it does:** The model makes its final decision (Real vs. Spoof).

* **Details:** The frame-level sequences $S$ and $F$ are mean-pooled into utterance-level vectors $s$ and $f$. They are concatenated with the holistic WavLM embedding to create a fused vector $g \in \mathbb{R}^{384}$. A final MLP classifier outputs the binary probability of the audio being a deepfake.

### Stage 5: Sharpness-Aware Minimization (SAM)

* **What it does:** Traditional training often memorizes the specific artifacts of the training data (falling into "sharp minima"). NSF-PINN is optimized using SAM, which calculates a worst-case weight perturbation $\epsilon^*$ at every step.

* **Details:** By optimizing for the worst-case loss within a neighborhood $\rho$, SAM forces the model to settle into "flat minima." This guarantees that when the model encounters slightly shifted distributions in the real world (e.g., WhatsApp audio compression or a brand-new AI voice generator), its performance remains stable.

## 🚀 Installation & Requirements

1. **Clone the repository:**

   ```
   git clone https://github.com/yourusername/NSF-PINN.git
   cd NSF-PINN
   
   ```

2. **Install dependencies:**
   Ensure you have a CUDA-capable GPU and the appropriate PyTorch version installed.

   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install transformers soundfile scipy scikit-learn tqdm
   sudo apt-get update && sudo apt-get install -y libsndfile1
   
   ```

3. **Download ASVspoof 2019 LA:**
   Download the official dataset and extract it into a `traindata/LA/` directory.

## 💻 Training

The training script includes an automatic workaround for PyTorch `GradScaler` state tracking during SAM's double-backward passes, making it stable for mixed-precision (AMP) training.

To train the model on a standard setup (e.g., 4GB VRAM GPU), use the following command:

```
python train.py \
  --data_root ./traindata/LA \
  --output_dir ./checkpoints \
  --batch_size 8 \
  --accum_steps 4 \
  --lr 3e-4 \
  --rho 0.05 \
  --frozen_layers 6 \
  --grad_checkpoint \
  --val_every 2 \
  --num_workers 1

```

## 📊 Evaluation

The script calculates the Equal Error Rate (EER) on the development set automatically. The training loop saves the best performing weights to `nsf_pinn_best.pt` in your designated output directory.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

