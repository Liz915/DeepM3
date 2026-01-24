import os
import torch
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

# ==========================================
# 🎯 配置区域 (Mac M1/M2/M3 最终稳定版)
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "checkpoints/dpo_qwen25_final"
DATA_FILE = "assets/dpo_dataset_final.jsonl"

# 训练配置
NUM_EPOCHS = 3
BATCH_SIZE = 1           # M1 显存吃紧，Batch Size 只能设 1
GRAD_ACCUM = 8           # 梯度累积，等效 Batch Size = 8
LEARNING_RATE = 1e-5     # DPO 标准学习率

def main():
    # 1. 设备检测
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Running DPO on {device.upper()} ...")

    # 2. 加载数据
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
    print(f"📚 Loaded {len(dataset)} valid samples.")

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 4. 加载模型
    # 使用 torch.float32 保证 MPS 绝对稳定 (0.5B 模型 FP32 也就 2GB 显存，M1 扛得住)
    print("🤖 Loading Model (FP32)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32, 
        device_map=None 
    ).to(device)

    # 5. LoRA 配置 (全量 Linear 层)
    # 补全 Qwen 的 MLP 层，效果更好
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
        use_dora=False # 关闭 DoRA，提升 MPS 训练速度
    )

    # 6. 训练参数
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        
        # 关闭混合精度，防止 MPS 报错 (速度慢点但能跑完)
        fp16=False,
        bf16=False,
        
        logging_steps=1,
        save_steps=50,
        report_to="tensorboard",
        remove_unused_columns=False,
        max_prompt_length=512,
        max_length=1024,
    )

    # 7. 初始化 Trainer
    print("🔥 Initializing DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # 显式设为 None，让 TRL 内部处理 Reference
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer, # 参数名已修正
        peft_config=peft_config,
    )

    # 8. 开始训练
    print("🏎️ Start Training! (This may take a while on M1...)")
    trainer.train()

    # 9. 保存
    print("💾 Saving LoRA adapter...")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()