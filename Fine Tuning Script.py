import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json

print("🚀 Starting Llama 3.2 1B Medical Fine-Tuning with LoRA")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATASET_PATH = os.path.join(os.getcwd(), "Dataset", "medical_chat_formatted.jsonl")
OUTPUT_DIR = os.path.join(os.getcwd(), "Fine Tuned Files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# LoRA Configuration
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# Training Configuration
TRAINING_CONFIG = {
    "num_train_epochs": 2,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 3e-4,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_steps": 250,
    "save_total_limit": 3,
    "fp16": False,
    "bf16": True,
    "optim": "paged_adamw_8bit",
    "max_seq_length": 1024,
}

print(f"📁 Dataset: {DATASET_PATH}")
print(f"📁 Output: {OUTPUT_DIR}")
print(f"🤖 Model: {MODEL_NAME}")
print(f"⚙️  LoRA Rank: {LORA_CONFIG['r']}")
print(f"📊 Epochs: {TRAINING_CONFIG['num_train_epochs']}")
print(f"📊 Effective Batch Size: {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
print("="*80)

# ============================================================================
# LOAD TOKENIZER
# ============================================================================

print("\n📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("✓ Tokenizer loaded")

# ============================================================================
# LOAD AND PREPARE DATASET
# ============================================================================

print("\n📥 Loading dataset...")

def load_jsonl_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return {"train": data}

# Load the dataset
raw_dataset = load_jsonl_dataset(DATASET_PATH)
print(f"✓ Loaded {len(raw_dataset['train'])} samples")

# Create train/eval split (95/5)
from datasets import Dataset
dataset = Dataset.from_list(raw_dataset['train'])
dataset = dataset.train_test_split(test_size=0.05, seed=42)

print(f"✓ Train samples: {len(dataset['train'])}")
print(f"✓ Eval samples: {len(dataset['test'])}")

# ============================================================================
# TOKENIZATION FUNCTION - FIXED FOR PADDING
# ============================================================================

def tokenize_function(examples):
    """
    Apply chat template and tokenize for Llama 3.2 1B Instruct
    """
    texts = []
    for messages in examples['messages']:
        # Apply the chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    
    # Tokenize with padding and truncation
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=TRAINING_CONFIG['max_seq_length'],
        padding="max_length",  # FIXED: Use max_length padding
        return_tensors=None,  # Return lists, not tensors
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

print("\n🔄 Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset['train'].column_names,
    desc="Tokenizing"
)
print("✓ Tokenization complete")

# ============================================================================
# LOAD MODEL WITH LORA
# ============================================================================

print("\n📥 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
print("✓ Model loaded")

# Apply LoRA
print("\n⚙️  Applying LoRA configuration...")
lora_config = LoraConfig(**LORA_CONFIG)
model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
print(f"✓ Total params: {total_params:,}")

# ============================================================================
# TRAINING SETUP - FIXED DATA COLLATOR
# ============================================================================

print("\n⚙️  Setting up training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=TRAINING_CONFIG['num_train_epochs'],
    per_device_train_batch_size=TRAINING_CONFIG['per_device_train_batch_size'],
    per_device_eval_batch_size=TRAINING_CONFIG['per_device_train_batch_size'],
    gradient_accumulation_steps=TRAINING_CONFIG['gradient_accumulation_steps'],
    learning_rate=TRAINING_CONFIG['learning_rate'],
    max_grad_norm=TRAINING_CONFIG['max_grad_norm'],
    warmup_ratio=TRAINING_CONFIG['warmup_ratio'],
    lr_scheduler_type=TRAINING_CONFIG['lr_scheduler_type'],
    logging_steps=TRAINING_CONFIG['logging_steps'],
    save_steps=TRAINING_CONFIG['save_steps'],
    save_total_limit=TRAINING_CONFIG['save_total_limit'],
    eval_strategy="steps",
    eval_steps=TRAINING_CONFIG['save_steps'],
    bf16=TRAINING_CONFIG['bf16'],
    fp16=TRAINING_CONFIG['fp16'],
    optim=TRAINING_CONFIG['optim'],
    gradient_checkpointing=True,
    group_by_length=False,  # FIXED: Disable to avoid length issues
    report_to="none",
    run_name="llama-3.2-1b-medical",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# FIXED: Simpler data collator that doesn't try to do MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,  # For efficiency
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    data_collator=data_collator,
)

print("✓ Training setup complete")

# ============================================================================
# START TRAINING
# ============================================================================

print("\n" + "="*80)
print("🚀 STARTING TRAINING")
print("="*80)
print(f"⏰ This will take approximately 30-60 minutes on H200")
print(f"📊 Training {len(tokenized_dataset['train'])} samples for {TRAINING_CONFIG['num_train_epochs']} epochs")
print("="*80 + "\n")

# Train
trainer.train()

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)

# ============================================================================
# SAVE MODEL
# ============================================================================

print("\n💾 Saving final model...")

# Save the LoRA adapter
final_model_path = os.path.join(OUTPUT_DIR, "final_model")
trainer.model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"✓ Model saved to: {final_model_path}")

# ============================================================================
# SAVE MERGED MODEL
# ============================================================================

print("\n💾 Saving merged model...")

from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load LoRA adapter
model_with_lora = PeftModel.from_pretrained(base_model, final_model_path)

# Merge and save
merged_model = model_with_lora.merge_and_unload()
merged_model_path = os.path.join(OUTPUT_DIR, "merged_model")
merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print(f"✓ Merged model saved to: {merged_model_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("🎉 FINE-TUNING COMPLETE!")
print("="*80)
print(f"📁 LoRA Adapter: {final_model_path}")
print(f"📁 Merged Model: {merged_model_path}")
print(f"✨ Model is ready for inference!")
print("="*80)

# ============================================================================
# QUICK TEST
# ============================================================================

print("\n🧪 Testing the model...")

test_messages = [
    {"role": "system", "content": "If you are a doctor, please answer the medical questions based on the patient's description."},
    {"role": "user", "content": "I have a severe headache and fever. What should I do?"}
]

# Prepare input
test_input = tokenizer.apply_chat_template(
    test_messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
).to(model.device)

# Generate
with torch.no_grad():
    output = merged_model.generate(
        test_input,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

# Decode
response = tokenizer.decode(output[0][test_input.shape[-1]:], skip_special_tokens=True)

print("\n" + "="*80)
print("Sample Output:")
print("="*80)
print(f"User: {test_messages[1]['content']}")
print(f"\nAssistant: {response}")
print("="*80)

print("\n✅ All done! Your medical AI is ready! 🏥")
