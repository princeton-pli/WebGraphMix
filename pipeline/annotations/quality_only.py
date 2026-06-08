"""Generate quality-only (DCLM-fasttext) annotation scores per document shard."""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import get_data_root


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DCLM-fasttext quality-only annotations.")
    parser.add_argument("--corpus-root", default=None, help="Corpus-200B root (default: $DATA_ROOT).")
    parser.add_argument(
        "--dclm-dir",
        default=None,
        help="Directory with precomputed DCLM-fasttext .npy scores (default: corpus_root/scores_dclm-fasttext).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dir (default: corpus_root/annotations_for_quality_based_document_sampling_dclmfilter).",
    )
    return parser.parse_args()


def generate_quality_annotations(corpus_root: str, dclm_dir: str, output_dir: str):
    docs_dir = os.path.join(corpus_root, "documents")
    os.makedirs(output_dir, exist_ok=True)

    dclm_files = sorted(glob.glob(os.path.join(dclm_dir, "*.npy")))
    print(f"Generating quality-based annotations for {len(dclm_files)} shards...")

    for dclm_path in dclm_files:
        shard_name = os.path.basename(dclm_path).replace(".npy", "")
        output_path = os.path.join(output_dir, f"{shard_name}.npy")
        doc_path = os.path.join(docs_dir, f"{shard_name}.jsonl.zst")

        if not os.path.exists(doc_path):
            print(f"Warning: No document shard found for {shard_name}, skipping.")
            continue
        if os.path.exists(output_path):
            print(f"Skipping {shard_name} (already exists)")
            continue

        try:
            quality_scores = np.load(dclm_path).astype(np.float32)
            np.save(output_path, quality_scores)
            print(f"Done: {shard_name} | {len(quality_scores):,} docs")
        except Exception as e:
            print(f"Error processing {shard_name}: {e}")

    print(f"\nFinished! Quality-only annotations saved to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    corpus_root = args.corpus_root or str(get_data_root())
    dclm_dir = args.dclm_dir or os.path.join(corpus_root, "scores_dclm-fasttext")
    output_dir = args.output_dir or os.path.join(
        corpus_root, "annotations_for_quality_based_document_sampling_dclmfilter"
    )
    generate_quality_annotations(corpus_root, dclm_dir, output_dir)
