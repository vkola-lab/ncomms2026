"""Run entropy_calc.py for all model / save_name pairs."""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENTROPY_CALC = os.path.join(SCRIPT_DIR, "entropy_generate.py")

MODELS = [
    (
        "Qwen/Qwen2.5-3B-Instruct",
        "../intermediate_files/entropies/q3b.json",
    ),
    (
        "Qwen/Qwen2.5-7B-Instruct",
        "../intermediate_files/entropies/q7b.json",
    ),
    (
        "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/ckpt/ckpt_access/qwen25_3B_drgrpo_gp16_nacc_inc_oversample_dedup",
        "../intermediate_files/entropies/oversample_dedup.json",
    ),
    (
        "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/ckpt/ckpt_access/qwen25_3B_drgrpo_gp16_nacc_inc_oversample_dedup_sce_tanh",
        "../intermediate_files/entropies/oversample_dedup_sce_tanh.json",
    ),
    (
        "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/ckpt/ckpt_access/qwen25_3B_drgrpo_gp16_nacc_inc_oversample",
        "../intermediate_files/entropies/oversample.json",
    ),
    (
        "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/ckpt/ckpt_access/qwen25_3B_drgrpo_gp16_nacc_inc_oversample_sce_tanh",
        "../intermediate_files/entropies/oversample_sce_tanh.json",
    ),
]


def main():
    os.makedirs(os.path.join(SCRIPT_DIR, "../intermediate_files/entropies"), exist_ok=True)

    for i, (model_id, save_name) in enumerate(MODELS, start=1):
        print("=" * 80)
        print(f"[{i}/{len(MODELS)}] model_id={model_id}")
        print(f"[{i}/{len(MODELS)}] save_name={save_name}")
        print("=" * 80)

        cmd = [
            sys.executable,
            ENTROPY_CALC,
            "--model_id",
            model_id,
            "--save_name",
            save_name,
        ]
        result = subprocess.run(cmd, cwd=SCRIPT_DIR)
        if result.returncode != 0:
            print(f"FAILED ({result.returncode}): {model_id} -> {save_name}")
            sys.exit(result.returncode)

        print(f"Done: {save_name}\n")

    print("All models finished.")


if __name__ == "__main__":
    main()
