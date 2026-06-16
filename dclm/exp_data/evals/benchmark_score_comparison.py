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

# Names match HuggingFace checkpoint folders (PrincetonPLI/WebGraphMix-openlm-1B).
files_to_process = {
    "evaluation_random_selection_mmlu_and_lowvar.json": "random baseline",
    "evaluation_dclm_fasttext_only_mmlu_and_lowvar.json": "quality (DCLM-fasttext)",
    "evaluation_betweenness_alpha0.5_mmlu_and_lowvar.json": "WebGraphMix 50/50",
    "evaluation_betweenness_alpha0.5_mult_div_dclm_fasttext_mmlu_and_lowvar.json": "WebGraphMix+ multiply 50/50",
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
