Medical Reasoning Model: Llama 3.2 1B Fine-Tuned on ReasonMed
Overview
This repository contains scripts and instructions for fine-tuning the Meta Llama 3.2 1B Instruct model on the ReasonMed dataset, merging the LoRA adapters, and converting the model to GGUF format for efficient deployment. The project focuses on creating a specialized AI for medical reasoning tasks, such as diagnosing symptoms or answering clinical questions. It leverages low-rank adaptation (LoRA) for parameter-efficient fine-tuning and quantization for optimized inference.
The ReasonMed dataset (arXiv:2506.09513) is the largest medical reasoning dataset available, with 370,000 high-quality examples generated via a multi-agent system using complementary LLMs. It was curated through an easy-medium-difficult (EMD) pipeline to ensure diverse and robust training data. This fine-tuning enhances the model's ability to perform chain-of-thought reasoning, where it breaks down problems step by step before providing a final answer.
This was a personal project funded with approximately 1500 INR to rent an H100 GPU from JarvisLabsAI, an Indian platform for affordable GPU access. It represents my first large-scale fine-tuning effort using personal resources.
Key Features

Base Model: Meta Llama 3.2 1B Instruct (causal language model optimized for instruction-following).
Fine-Tuning Method: LoRA with rank=8, alpha=16, targeting query, key, value, and output projection modules.
Dataset: ReasonMed (370k examples in JSONL format, focused on medical conversations).
Training Setup: 3 epochs, batch size=4, gradient accumulation=4, learning rate=2e-4, mixed precision (FP16).
Merging: PEFT-based merging of LoRA adapters into the base model.
Conversion: GGUF format via llama.cpp, with quantizations (Q4_K_M, Q5_K_M, Q8_0) for different trade-offs in size, speed, and quality.
Observed Improvements: Post-fine-tuning, the model exhibits step-by-step reasoning, improving accuracy on medical queries (e.g., symptom diagnosis).

Prerequisites

Python 3.10+ with libraries: transformers, peft, datasets, torch, accelerate.
GPU with CUDA support (e.g., H100 for training; compatible hardware for inference).
Git for cloning llama.cpp.
Access to the ReasonMed dataset (download from the arXiv paper's linked repository).
Hugging Face account for model loading (requires authentication for Llama models).

Install dependencies:
textpip install torch transformers peft datasets accelerate
Directory Structure

fine_tune.py: Script for LoRA fine-tuning on the dataset.
merge.py: Script to merge LoRA adapters into the base model.
convert_to_gguf.py: Script for converting the merged model to GGUF and quantizing it.
Dataset/: Place for reasonmed_1000.jsonl (subset; scale to full 370k as needed).
Output/: Directory for saved LoRA adapters.
Merged/: Directory for the merged model.
GGUF/: Directory for GGUF files (e.g., llama3.2-1b-medical-reasonmed-Q4_K_M.gguf).

Usage Instructions
1. Fine-Tuning
Run fine_tune.py to train the model:
textpython fine_tune.py

Loads the base model and tokenizer.
Applies LoRA configuration.
Processes the dataset: Formats conversations using chat templates, tokenizes to max length 512.
Trains with Trainer API, saving adapters to /root/Output (adjust paths as needed).
Includes a quick inference test on a sample medical question.

2. Merging LoRA Adapters
Run merge.py to integrate adapters:
textpython merge.py

Loads the base model and adapters from Output/.
Merges using PEFT's merge_and_unload().
Saves the unified model to Merged/.
Performs a test inference on a sample query.

3. GGUF Conversion and Quantization
Run convert_to_gguf.py:
textpython convert_to_gguf.py

Clones llama.cpp if not present.
Converts merged model to FP16 GGUF.
Builds quantization tool.
Generates quantized versions: Q4_K_M (~800MB, balanced), Q5_K_M (~1GB, higher quality), Q8_0 (~1.3GB, highest quality).
Outputs to GGUF/.

Example Inference
After conversion, use tools like llama.cpp or Ollama for inference. For example, with llama.cpp:
text./llama.cpp/main -m GGUF/llama3.2-1b-medical-reasonmed-Q4_K_M.gguf --prompt "A patient with severe headache and neck stiffness. Likely diagnosis?"
Expected output: Step-by-step reasoning leading to "Meningitis" (option B in sample).
Performance Notes

Training Time: Approximately 1-2 hours on H100 GPU for full dataset (adjust batch size for memory).
Model Size: Base ~2GB; quantized GGUF versions reduce to 800MB-1.3GB.
Quality: Q4_K_M recommended for most use cases; test on medical benchmarks like PubMedQA for validation.
Limitations: Model is 1B parameters, so best for lightweight applications. For production, evaluate on diverse medical data to avoid biases.

Citation
If using this project, cite the ReasonMed paper:
text@article{sun2025reasonmed,
  title={ReasonMed: A 370K Multi-Agent Generated Dataset for Advancing Medical Reasoning},
  author={Sun, Yu and Qian, Xingyu and Xu, Weiwen and Zhang, Hao and Xiao, Chenghao and Li, Long and Zhao, Deli and Huang, Wenbing and Xu, Tingyang and Bai, Qifeng and Rong, Yu},
  journal={arXiv preprint arXiv:2506.09513},
  year={2025}
}
License
This project is licensed under the MIT License. The base Llama model follows Meta's license terms. ReasonMed dataset usage should comply with its terms.
Acknowledgments

Meta for Llama models.
Authors of ReasonMed for the dataset.
JarvisLabsAI for GPU rental.
Hugging Face and PEFT for fine-tuning tools.
ggerganov/llama.cpp for GGUF conversion.

For questions or contributions, open an issue or contact me via LinkedIn.
