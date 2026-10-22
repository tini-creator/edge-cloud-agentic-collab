import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Bulletproof Pathing
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SFT_ADAPTER_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models", "sft-ue-baseline"))
MERGED_MODEL_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models", "sft-ue-merged"))
BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"

print(f"Looking for SFT adapter at: {SFT_ADAPTER_DIR}")
print(f"Will save merged model to: {MERGED_MODEL_DIR}")

# 2. Load Base Model
print("\nLoading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map={"": "cpu"},
    dtype=torch.float32
)

# 3. Merge and Unload
print("Loading SFT adapter and merging into base weights...")
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_DIR)
merged_model = model.merge_and_unload()

# 4. Save the new unified model
print("Saving merged model...")
merged_model.save_pretrained(MERGED_MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.save_pretrained(MERGED_MODEL_DIR)

print("\nMerge complete! The folder 'models/sft-ue-merged' has been created.")