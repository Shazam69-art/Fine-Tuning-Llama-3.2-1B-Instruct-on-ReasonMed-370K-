# ReasonMed-LoRA-1B: Medical Reasoning Model Fine-Tuned on ReasonMed

## Overview

This repository contains the complete pipeline for fine-tuning Meta's **Llama-3.2-1B-Instruct** model on the **ReasonMed** dataset (370k examples) using LoRA, merging the adapters, and converting the result into GGUF format for efficient local inference.

The fine-tuned model demonstrates clear step-by-step (Chain-of-Thought) medical reasoning before arriving at final answers — a behavior that emerges naturally after training on this high-quality reasoning dataset.

**Key Features**
- Parameter-efficient fine-tuning with LoRA (only ~0.1–0.3% of parameters updated)
- Full support for chat-template conversations from ReasonMed
- Mixed-precision training (FP16) optimized for modern GPUs
- Merged model + GGUF quantization (Q4_K_M, Q5_K_M, Q8_0)
- Lightweight inference-ready files (~800 MB – 1.3 GB)
- Observable CoT reasoning on medical multiple-choice and open-ended questions

**Important Disclaimer**  
This model is for **research, education, and prototyping purposes only**. It is **not** a medical device, diagnostic tool, or substitute for professional clinical judgment. Always consult qualified healthcare professionals for medical decisions.

## Dataset

**ReasonMed** — the largest publicly available medical reasoning dataset (as of 2025)  
- **Paper**: [ReasonMed: A 370K Multi-Agent Generated Dataset for Advancing Medical Reasoning](https://arxiv.org/abs/2506.09513) (arXiv:2506.09513v3)  
- **Size**: 370,000 high-quality reasoning examples  
- **Generation**: Multi-agent LLM pipeline + Error Refiner + easy-medium-difficult (EMD) curation  
- **Format**: JSONL with `conversations` containing role-based turns (system, user, assistant)  
- **License**: Follows the terms set by the ReasonMed authors (check repository for details)

## Architecture & Training

### Fine-Tuning
- **Base model**: `meta-llama/Llama-3.2-1B-Instruct`  
- **Method**: LoRA (rank=8, alpha=16, dropout=0.05)  
- **Target modules**: q_proj, k_proj, v_proj, o_proj  
- **Optimizer**: Adafactor  
- **Hyperparameters**:  
  - epochs: 3  
  - global batch size: 16 (per-device 4 + gradient accumulation 4)  
  - learning rate: 2e-4  
  - warmup steps: 20  
  - max sequence length: 512  
- **Hardware used**: NVIDIA H100 (rented via JarvisLabs.ai)

### Post-Training Steps
1. Merge LoRA adapters into the base model (PEFT `merge_and_unload`)  
2. Convert merged model → GGUF (FP16 intermediate)  
3. Quantize to Q4_K_M (recommended), Q5_K_M, Q8_0 using llama.cpp

## Requirements

**Training / Merging**
- Python 3.10+
- `transformers`, `peft`, `datasets`, `torch`, `accelerate`
- GPU with ≥24 GB VRAM (A100/H100 recommended)

**GGUF Conversion**
- `llama.cpp` repository (cloned automatically in script)
- `cmake`, `make` (for building quantize binary)

**Inference**
- `llama.cpp`, Ollama, LM Studio, KoboldCPP, or any GGUF-compatible runner

## Usage

### 1. Fine-Tuning
```bash
# Adjust DATASET_PATH and OUTPUT_DIR in the script if needed
python fine_tune.py
2. Merge LoRA → Full Model
Bashpython merge.py
3. Convert to GGUF + Quantize
Bashpython convert_to_gguf.py
Output files will appear in the GGUF/ folder:

llama3.2-1b-medical-reasonmed-fp16.gguf
llama3.2-1b-medical-reasonmed-Q4_K_M.gguf  (~800 MB – best balance)
llama3.2-1b-medical-reasonmed-Q5_K_M.gguf
llama3.2-1b-medical-reasonmed-Q8_0.gguf

4. Run Inference (example with llama.cpp)
Bash./llama.cpp/main \
  -m GGUF/llama3.2-1b-medical-reasonmed-Q4_K_M.gguf \
  --color --temp 0.7 --top-p 0.9 \
  -p "A patient presents with fever, cough, and shortness of breath. What is the most appropriate initial investigation?\nA. ECG\nB. Chest X-ray\nC. Blood culture\nD. CT pulmonary angiogram"
Project Background
This was my first personally funded large-scale fine-tuning project.

Spent ~₹1,500 renting an H100 GPU from JarvisLabs.ai (Indian cloud provider)
Trained on the full ReasonMed 370k dataset
Goal: Create an accessible, local-run medical reasoning model with strong CoT behavior

Limitations & Future Work

1B-parameter base → best suited for lightweight / edge use cases
Reasoning quality still lags behind 7B–70B medical models
No instruction-tuning / preference optimization performed yet
Possible next steps:
DPO / ORPO alignment
Fine-tuning larger base (Llama-3.2-3B, Meditron, etc.)
Evaluation on MedQA, PubMedQA, MMLU-clinical subsets


License

Code in this repository: MIT
Base model: Follows Meta Llama 3.2 Community License
Fine-tuned weights & GGUF files: Same as base model + dataset terms
ReasonMed dataset: Follow terms from the official repository

Acknowledgments

ReasonMed authors (Yu Sun et al.) for releasing the 370k dataset
Meta for open-weight Llama-3.2 models
JarvisLabs.ai for affordable H100 access in India
Hugging Face (transformers + PEFT)
Georgi Gerganov and llama.cpp contributors

Questions, suggestions, or want to collaborate?
→ Open an issue or reach out via LinkedIn.
