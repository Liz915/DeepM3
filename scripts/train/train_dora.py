import sys
import os
import torch
import yaml
from peft import LoraConfig, get_peft_model, TaskType

sys.path.append(os.getcwd())
from src.dynamics.modeling import DeepM3Model

def train_dora():
    # 1. Load Base Model
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    #  n_items
    model = DeepM3Model(cfg, n_items=3707)
    
    # 2. Configure DoRA
    # DoRA  use_dora=True (PEFT >= 0.9.0)
    #  target_modules
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        #  GRUCell Embedding
        # Embedding  modules_to_save  target_modules=['head']
        target_modules=["head"], 
        #  embedding item_emb  modules_to_save
        modules_to_save=["item_emb"], 
        use_dora=True
    )
    
    # 3. Apply Adapter
    print(" Applying DoRA adapters...")
    dora_model = get_peft_model(model, peft_config)
    dora_model.print_trainable_parameters()
    
    # 4. Dummy Training Loop ()
    optimizer = torch.optim.AdamW(dora_model.parameters(), lr=1e-4)
    print(" DoRA Model ready for training.")
    # ... standard training loop ...
    
    # 5. Save Adapter
    dora_model.save_pretrained("checkpoints/dora_adapter")
    print(" DoRA adapter saved.")

if __name__ == "__main__":
    train_dora()