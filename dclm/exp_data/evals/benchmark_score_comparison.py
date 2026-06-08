import json
import os
import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Eval JSON files live in this directory (run experiments/eval/mmlu_and_lowvar.sh first).
EVAL_DIR = os.environ.get(
    "EVAL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__))),
)

# Example paths (relative to EVAL_DIR):
#   evaluation_baseline_random_corpus_32b-open_lm_1b_swiglutorch-..._mmlu_and_lowvar.json
#   evaluation_betweenness_50top_corpus_32b-open_lm_1b_swiglutorch-..._mmlu_and_lowvar.json
#   evaluation_centralitydclmfiltermultiply_betweenness_50top_corpus_32b-open_lm_1b_swiglutorch-..._mmlu_and_lowvar.json
#
# fill in your files here — map filename -> short column label for the comparison sheet:
files_to_process = {
    "evaluation_baseline_random_corpus_32b-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000_mmlu_and_lowvar.json": "baseline",
    "evaluation_quality_only_dclmfilter_corpus_32b-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000_mmlu_and_lowvar.json": "quality",
    "evaluation_betweenness_50top_corpus_32b-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000_mmlu_and_lowvar.json": "betweenness 50top",
    "evaluation_centralitydclmfiltermultiply_betweenness_50top_corpus_32b-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000_mmlu_and_lowvar.json": "multiply betweenness 50top",
    "evaluation_centralitydclmfilteradd_betweenness_subtract_25top_corpus_32b-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000_mmlu_and_lowvar.json": "add subtract betweenness 25top",
    "evaluation_centralitydclmfilteradd_betweenness_subtract_50top_corpus_32b-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000_mmlu_and_lowvar.json": "add subtract betweenness 50top",
    "evaluation_centralitydclmfilteradd_betweenness_subtract_75top_corpus_32b-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000_mmlu_and_lowvar.json": "add subtract betweenness 75top",
    "evaluation_centralitydclmfilteradd_betweenness_subtract_bottomk_corpus_32b-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000_mmlu_and_lowvar.json": "add subtract betweenness bottomk",
    "evaluation_centralitydclmfilteradd_betweenness_topk_corpus_32b-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000_mmlu_and_lowvar.json": "add betweenness topk",
    "evaluation_centralitydclmfiltermultiply_betweenness_divide_25top_corpus_32b-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000_mmlu_and_lowvar.json": "multiply divide betweenness 25top",
    "evaluation_centralitydclmfiltermultiply_betweenness_divide_50top_corpus_32b-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000_mmlu_and_lowvar.json": "multiply divide betweenness 50top",
    "evaluation_centralitydclmfiltermultiply_betweenness_divide_75top_corpus_32b-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000_mmlu_and_lowvar.json": "multiply divide betweenness 75top",
    "evaluation_centralitydclmfiltermultiply_betweenness_divide_bottomk_corpus_32b-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000_mmlu_and_lowvar.json": "multiply divide betweenness bottomk",
    "evaluation_centralitydclmfiltermultiply_betweenness_topk_corpus_32b-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000_mmlu_and_lowvar.json": "multiply betweenness topk",
}

OUTPUT_FILENAME = os.environ.get("OUTPUT_FILENAME", "scores.xlsx")


def main():
    collected_data = {}
    print("--- Starting Data Extraction ---")

    for filename, header_name in files_to_process.items():
        filepath = os.path.join(EVAL_DIR, filename)
        if not os.path.exists(filepath):
            print(f"WARNING: File not found: {filepath}")
            continue

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            if "eval_metrics" in data and "icl" in data["eval_metrics"]:
                collected_data[header_name] = data["eval_metrics"]["icl"]
                print(f"Loaded: {header_name}")
            else:
                print(f"Error: Structure not found in {filepath}")

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    df = pd.DataFrame(collected_data)
    model_columns = [h for h in files_to_process.values() if h in df.columns]
    if not model_columns:
        print("No evaluation files found. Run experiments/eval/mmlu_and_lowvar.sh first.")
        return
    df = df[model_columns]
    df["Average"] = df.mean(axis=1, numeric_only=True)
    df.index.name = "Test"
    df.reset_index(inplace=True)

    styler = df.style.highlight_max(
        subset=model_columns,
        axis=1,
        props="font-weight: bold; background-color: #FFFF00;",
    )
    styler = styler.format(precision=4)

    try:
        styler.to_excel(OUTPUT_FILENAME, index=False, engine="openpyxl")
        print(f"Success! File generated: {OUTPUT_FILENAME}")
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == "__main__":
    main()
