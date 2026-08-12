# LlamaTron RS1 ThinkDoc

A medical conversation model built by fine-tuning Llama 3.2 1B Instruct on 112K real doctor-patient consultations — the first release in a progressive fine-tuning series scaling toward multi-million-sample training.

[![License](https://img.shields.io/badge/License-Llama%203.2%20Community-38A169?style=flat-square)](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
[![Base Model](https://img.shields.io/badge/Base-Llama%203.2%201B%20Instruct-4A5568?style=flat-square)](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
[![Dataset](https://img.shields.io/badge/Dataset-112K%20conversations-2B6CB0?style=flat-square)](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k)
[![Method](https://img.shields.io/badge/Method-LoRA-DB2777?style=flat-square)](https://github.com/sufirumii/LlamaTron-RS1-ThinkDoc)

## Overview

LlamaTron RS1 ThinkDoc represents the first model in a progressive fine-tuning series, beginning with 100K-range datasets and scaling toward multi-million sample training in future iterations. This model demonstrates human-aligned conversational capabilities in medical consultation contexts, trained on real doctor-patient interactions rather than synthetic reasoning traces.

## Architecture

<img width="1800" height="1200" alt="image" src="https://github.com/user-attachments/assets/2e8f12ae-d926-45b2-9078-cc7835637541" />


Training produces two usable artifacts from the same run: the LoRA adapter alone (`final_model/`, small and easy to swap between base models) and a fully merged checkpoint (`merged_model/`, no PEFT dependency needed at inference time). The published Hugging Face model uses the merged version.

## Model Details

### Base model
- **Architecture**: Llama 3.2 1B Instruct (meta-llama/Llama-3.2-1B-Instruct)
- **Parameters**: 1.24 billion total parameters
- **Developer**: Meta AI

### Fine-tuning specifications
- **Method**: LoRA (Low-Rank Adaptation)
- **LoRA rank**: 16
- **LoRA alpha**: 32
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Trainable parameters**: 11.2M (0.90% of total)
- **Dropout**: 0.05

## Dataset

- **Name**: ChatDoctor-HealthCareMagic-100k
- **Source**: [https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k)
- **Size**: 112,165 medical conversation samples
- **Structure**: Instruction-input-output format containing real doctor-patient interactions
- **Domain**: General medical consultation across various specialties

## Training Configuration

### Hardware
- **GPU**: NVIDIA H200
- **Training time**: Approximately 3 hours

### Hyperparameters
- **Epochs**: 2
- **Batch size per device**: 4
- **Gradient accumulation steps**: 4
- **Effective batch size**: 16
- **Learning rate**: 3e-4
- **Learning rate scheduler**: Cosine with 3% warmup
- **Optimizer**: Paged AdamW 8-bit
- **Max sequence length**: 1024 tokens
- **Precision**: BFloat16

### Data split
- **Training**: 106,548 samples (95%)
- **Evaluation**: 5,608 samples (5%)

## Model Capabilities

The model is designed to:
- Generate medically-informed responses to patient queries
- Maintain professional and empathetic communication style
- Provide structured medical advice including symptom analysis and recommendations
- Mirror the consultation patterns of practicing physicians

## Limitations

- Trained exclusively on text-based medical conversations
- Should not replace professional medical advice
- Limited to patterns observed in training data
- May generate plausible but incorrect medical information
- Not evaluated on clinical benchmarks

## Technical Stack

- **Framework**: Hugging Face Transformers
- **Fine-tuning library**: PEFT (Parameter-Efficient Fine-Tuning)
- **Training library**: Hugging Face Trainer
- **Model format**: SafeTensors

## File Structure
```
LlamaTron-RS1-ThinkDoc/
├── Dataset/
│   └── medical_chat_formatted.jsonl
├── Fine Tuned Files/
│   ├── final_model/          # LoRA adapter weights
│   └── merged_model/         # Full merged model
└── scripts/
    ├── dataset_preparation.py
    ├── training.py
    └── inference_interface.py
```

## Usage

First, authenticate with the Hugging Face Hub:

```bash
hf auth login
```

**Using a pipeline (high-level helper):**

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="Rumiii/LlamaTron-RS1-ThinkDoc")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)
```

**Loading the model directly:**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Rumiii/LlamaTron-RS1-ThinkDoc")
model = AutoModelForCausalLM.from_pretrained("Rumiii/LlamaTron-RS1-ThinkDoc")

messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
```

## Future Work

### RS2 development
- Integration of 800K+ chain-of-thought reasoning samples
- Extended context length support
- Multi-turn conversation enhancement

### Long-term roadmap
- Scaling to multi-million sample datasets
- Clinical benchmark evaluation
- Multilingual medical consultation support
- Specialized domain adaptations

## Ethical Considerations

This model is intended for research and educational purposes. Medical advice generated by this AI should not be considered a substitute for professional medical consultation. Users should always consult qualified healthcare providers for medical decisions.

## Acknowledgments

- Meta AI for the Llama 3.2 base model
- Lavita for the ChatDoctor-HealthCareMagic-100k dataset
- Hugging Face for the transformers and PEFT libraries

## Author

Rumi Iqbal Sufi
AI Engineer
GitHub: https://github.com/sufirumii
Hugging Face: https://huggingface.co/Rumiii
LinkedIn: https://www.linkedin.com/in/rumi-sufi-6323a5265/

## License

This project uses the Llama 3.2 model, which is subject to Meta's license terms. The fine-tuned weights are released under the same license. Please refer to the original Llama 3.2 license for usage terms.

---

**Version**: RS1 (Research Series 1)
**Status**: Completed
**Last updated**: February 2026
