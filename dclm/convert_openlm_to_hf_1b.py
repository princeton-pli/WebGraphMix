import os
import glob
import json
import torch
import gc
import re
from transformers import AutoTokenizer

# Import the custom OpenLM architecture wrappers
import open_lm
from open_lm.hf.configuration_openlm import OpenLMConfig
from open_lm.hf.modeling_openlm import OpenLMForCausalLM

# ==========================================
# 1. SETUP PATHS
# ==========================================
REPO_ROOT = os.environ.get("REPO_ROOT", os.path.dirname(os.path.abspath(__file__)))
INPUT_ROOT = os.environ.get("CHECKPOINT_INPUT_DIR", os.path.join(REPO_ROOT, "checkpoints_to_convert"))
OUTPUT_ROOT = os.environ.get("CHECKPOINT_HF_OUTPUT_DIR", os.path.join(REPO_ROOT, "checkpoints_hf"))

open_lm_install_dir = os.path.dirname(open_lm.__file__)
CONFIG_PATH = os.path.join(open_lm_install_dir, "model_configs", "open_lm_1b.json")

# ==========================================
# 2. INITIALIZE HUGGING FACE SETTINGS
# ==========================================
OpenLMConfig.register_for_auto_class()
OpenLMForCausalLM.register_for_auto_class("AutoModelForCausalLM")

print("Loading GPT-NeoX tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

print(f"Loading base blueprint from:\n{CONFIG_PATH}")
with open(CONFIG_PATH, "r") as f:
    base_config_dict = json.load(f)

# ==========================================
# 3. LOOP THROUGH ALL MODELS
# ==========================================
model_folders = [f for f in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, f))]

for folder_name in model_folders:
    print(f"\n" + "="*60)
    print(f"Processing: {folder_name}")
    print("="*60)

    level1_path = os.path.join(INPUT_ROOT, folder_name)
    out_dir = os.path.join(OUTPUT_ROOT, folder_name)

    subdirs = [d for d in os.listdir(level1_path) if os.path.isdir(os.path.join(level1_path, d))]
    if not subdirs:
        print(f"SKIPPING: No secondary folder found inside {level1_path}")
        continue

    secondary_folder_name = subdirs[0]
    level2_path = os.path.join(level1_path, secondary_folder_name)
    checkpoint_dir = os.path.join(level2_path, "checkpoints")

    pt_files = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt"))
    if not pt_files:
        print(f"SKIPPING: Could not find any epoch_*.pt file in {checkpoint_dir}")
        continue

    pt_files.sort(key=lambda x: int(re.search(r"epoch_(\d+)\.pt", x).group(1)) if re.search(r"epoch_(\d+)\.pt", x) else 0)
    checkpoint_path = pt_files[-1]
    print(f"Using checkpoint: {os.path.basename(checkpoint_path)}")

    try:
        # 1. Load raw weights
        print("Loading PyTorch weights into memory (CPU)...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        raw_state_dict = checkpoint.get("state_dict", checkpoint)

        # 2. Start from 1B base config and override dims from the actual weights
        config_dict = base_config_dict.copy()

        # Measure Vocab Size and Hidden Dimension
        embed_key = next(
            (k for k in raw_state_dict if k.replace("_orig_mod.", "").replace("module.", "") == "tok_embeddings.weight"),
            None
        )
        if embed_key is not None:
            vocab_size, hidden_dim = raw_state_dict[embed_key].shape
            config_dict["vocab_size"] = vocab_size
            config_dict["dim"] = hidden_dim

        # Count the number of layers
        layer_keys = [k for k in raw_state_dict.keys() if "layers." in k]
        if layer_keys:
            max_layer = max([int(re.search(r"layers\.(\d+)\.", k).group(1)) for k in layer_keys if re.search(r"layers\.(\d+)\.", k)])
            config_dict["n_layers"] = max_layer + 1

        # For 1B models the folder name uses open_lm_1b style (no _h=N suffix),
        # so n_heads comes from the base config (16 for 1B). Override only if present.
        heads_match = re.search(r'_h=(\d+)', secondary_folder_name)
        if heads_match:
            config_dict["n_heads"] = int(heads_match.group(1))

        # Parse training params.txt to recover critical architectural settings that
        # are NOT in the base open_lm JSON config but ARE needed for correct inference.
        # Without these, the model runs with wrong norms and missing QK normalization,
        # producing near-random (chance-level) eval scores.
        params_txt_path = os.path.join(level2_path, "params.txt")
        if os.path.exists(params_txt_path):
            with open(params_txt_path, "r") as pf:
                for line in pf:
                    line = line.strip()
                    if line.startswith("model_norm:"):
                        config_dict["norm_type"] = line.split(":", 1)[1].strip()
                    elif line.startswith("qk_norm:"):
                        val = line.split(":", 1)[1].strip()
                        config_dict["apply_qk_norm"] = (val.lower() == "true")
                    elif line.startswith("attn_name:"):
                        config_dict["attn_name"] = line.split(":", 1)[1].strip()
                    elif line.startswith("ffn_type:"):
                        config_dict["ffn_type"] = line.split(":", 1)[1].strip()
            print(f"From params.txt: norm_type={config_dict.get('norm_type')}, apply_qk_norm={config_dict.get('apply_qk_norm')}, attn_name={config_dict.get('attn_name')}")
        else:
            # Fallback: infer from checkpoint structure
            qk_keys = [k for k in raw_state_dict.keys() if "q_norm" in k or "k_norm" in k]
            if qk_keys:
                config_dict["apply_qk_norm"] = True
                print(f"No params.txt found; inferred apply_qk_norm=True from checkpoint ({len(qk_keys)} q/k_norm keys)")
            has_norm_bias = any("attention_norm.bias" in k or "ffn_norm.bias" in k for k in raw_state_dict.keys())
            if not has_norm_bias:
                config_dict["norm_type"] = "gain_only_lp_layer_norm"
                print(f"No params.txt found; inferred norm_type=gain_only_lp_layer_norm (no bias in checkpoint norms)")

        print(f"Architecture: dim={config_dict.get('dim')}, layers={config_dict.get('n_layers')}, heads={config_dict.get('n_heads')}")

        # 3. Create the correctly-configured HF model shell
        config = OpenLMConfig(**config_dict)
        model = OpenLMForCausalLM(config)

        # 4. Clean checkpoint key prefixes.
        # FSDP checkpoints have clean keys already, but strip module./_orig_mod.
        # defensively in case of DDP or torch.compile artifacts.
        clean_state_dict = {}
        for key, value in raw_state_dict.items():
            new_key = key.replace("_orig_mod.", "").replace("module.", "")
            clean_state_dict[new_key] = value

        # 5. Load weights directly into model.model (the raw Transformer).
        # inv_freq is a RoPE buffer not saved in checkpoints — ignore it in unexpected keys.
        print("Transferring weights into Hugging Face wrapper...")
        load_result = model.model.load_state_dict(clean_state_dict, strict=False)
        if load_result.missing_keys:
            print(f"Missing keys (in model but not checkpoint): {load_result.missing_keys[:10]}")
        unexpected = [k for k in load_result.unexpected_keys if "inv_freq" not in k]
        if unexpected:
            print(f"Unexpected keys (in checkpoint but not model): {unexpected[:10]}")

        # 6. Save to HF format
        print(f"Saving Hugging Face model to {out_dir}...")
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

        print(f"SUCCESSFULLY CONVERTED: {folder_name}")

    except Exception as e:
        print(f"ERROR while converting {folder_name}: {e}")

    finally:
        if 'checkpoint' in locals(): del checkpoint
        if 'raw_state_dict' in locals(): del raw_state_dict
        if 'clean_state_dict' in locals(): del clean_state_dict
        if 'model' in locals(): del model
        gc.collect()

print(f"\nALL DONE! Your converted 1B models in:\n{OUTPUT_ROOT}")
