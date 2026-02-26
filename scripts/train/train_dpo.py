import os
import torch
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

# ==========================================
#   (Mac M1/M2/M3 )
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "checkpoints/dpo_qwen25_final"
DATA_FILE = "assets/dpo_dataset_final.jsonl"

# 
NUM_EPOCHS = 3
BATCH_SIZE = 1           # M1 Batch Size  1
GRAD_ACCUM = 8           #  Batch Size = 8
LEARNING_RATE = 1e-5     # DPO 

def main():
    # 1. 
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f" Running DPO on {device.upper()} ...")

    # 2. 
    if not os.path.exists(DATA_FILE):
         raise FileNotFoundError(f"Data file {DATA_FILE} not found!")
    
    data_list = []
    with open(DATA_FILE, "r") as f:
        for line in f:
            try:
                row = json.loads(line)
                if "prompt" in row and "chosen" in row and "rejected" in row:
                    data_list.append(row)
            except: pass

    dataset = Dataset.from_list(data_list)
    print(f" Loaded {len(dataset)} valid samples.")

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 4. 
    #  torch.float32  MPS  (0.5B  FP32  2GB M1 )
    print(" Loading Model (FP32)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32, 
        device_map=None 
    ).to(device)

    # 5. LoRA  ( Linear )
    #  Qwen  MLP 
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        use_dora=False #  DoRA MPS 
    )

    # 6. 
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        
        #  MPS  ()
        fp16=False,
        bf16=False,
        
        logging_steps=1,
        save_steps=50,
        report_to="tensorboard",
        remove_unused_columns=False,
        max_prompt_length=512,
        max_length=1024,
    )

    # 7.  Trainer
    print(" Initializing DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None, #  None TRL  Reference
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer, # 
        peft_config=peft_config,
    )

    # 8. 
    print(" Start Training! (This may take a while on M1...)")
    trainer.train()

    # 9. 
    print(" Saving LoRA adapter...")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()