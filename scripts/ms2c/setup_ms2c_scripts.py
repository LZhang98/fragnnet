import argparse
import os
import subprocess

SCAFFOLD_MODEL_TYPE_TO_IDS = {
    "fraggnn_d4_long": ["wpf5lx0s", "i9sbcq1x", "e4ps82x2", "654110tn", "udv8nild"],
    "neims_long": ["f86qx5mc", "n78taefc", "j6l4pig7", "nlmxw6uk", "ihtfbuyh"],
    "massformer_long": ["wbcacs9a", "9goj2crn", "mfinb7je", "xryfrdrt", "n8x5946i"],
}

INCHIKEY_MODEL_TYPE_TO_IDS = {
    "fraggnn_d4_long": ["okoezseo", "k6pjarvc", "njwchu0d", "euspkn0w", "z7r2i8pc"],
    "neims_long": ["xclueecr", "68wj70ah", "px0nb7mk", "aio4nee7", "0467epvx"],
    "massformer_long": ["i75p4rqv", "dey5m0xz", "luyzp00n", "muzcg5hy", "81vgna81"],
}


def main(args):
    QOS = args.qos
    GPU_TYPE = args.gpu_type

    for model_type in args.model_types:
        BATCH_SIZE = 1 if model_type == "fraggnn_d4_long" else 32

        for split_type in args.split_types:
            SPLIT = split_type
            DIR = args.ms2c_dp
            MODEL = model_type
            BASENAME = f"ms2pubchem_nist20_d4_wr00_{SPLIT}_test_10ppm_MORGAN-R2_50"
            PROC_DP = f"{DIR}/proc/pubchem_nist20_d4_wr00_{SPLIT}_test_10ppm_MORGAN-R2_50"
            SAVE_DP = f"{DIR}/predicted/{BASENAME}"

            # Create the directory if it doesn't exist
            os.makedirs(SAVE_DP, exist_ok=True)

            FRAG_DP = (
                f"{DIR}/frags/d4_h4_isoFalse/pubchem_nist20_d4_wr00_{SPLIT}_test_10ppm_MORGAN-R2_50"
            )

            if SPLIT == "scaffold":
                MODEL_TYPE_TO_ID = SCAFFOLD_MODEL_TYPE_TO_IDS
            else:
                MODEL_TYPE_TO_ID = INCHIKEY_MODEL_TYPE_TO_IDS

            for seed_idx in range(args.num_seeds):
                SEED_IDX = seed_idx
                CUSTOM_FP = f"config/nist20v3-1_d4_wr00_{SPLIT}/{MODEL}/s{SEED_IDX}.yml"
                SAVE_FP = f"{SAVE_DP}/{MODEL}_s{SEED_IDX}.pkl"
                ID = MODEL_TYPE_TO_ID[MODEL][SEED_IDX]
                PY_FP = "scripts/ms2c/run_ms2c_predict_val.py"
                PY_ARGS = [
                    f"proc_dp={PROC_DP}",
                    f"frag_dp={FRAG_DP}",
                    f"wandb_run_id={ID}",
                    f"custom_fp={CUSTOM_FP}",
                    f"save_fp={SAVE_FP}",
                    f"batch_size={BATCH_SIZE}",
                    "num_workers=7",
                ]
                SETUP_CMD = [
                    "python",
                    "scripts/setup_job_vv.py",
                    "--python_script_fp",
                    PY_FP,
                    "--script_kwargs",
                    *PY_ARGS,
                    "--mem",
                    "64",
                    "--cpu",
                    "8",
                    "--job_name",
                    f"ms2c_{SPLIT}_{MODEL}_s{SEED_IDX}",
                    "--gpu_type",
                    GPU_TYPE,
                    "--qos",
                    QOS,
                ]
                print(SETUP_CMD)
                result = subprocess.run(SETUP_CMD, capture_output=True, text=True)
                print(result.stdout)
                print(result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--script_dp", type=str, required=True)
    parser.add_argument(
        "--model_types",
        type=str,
        nargs="+",
        default=[
            "fraggnn_d3_long",
            "fraggnn_d4_long",
            "neims_long",
            "massformer_long",
        ],
    )
    parser.add_argument(
        "--split_types",
        type=str,
        nargs="+",
        default=["inchikey", "scaffold"],
        choices=["inchikey", "scaffold"],
    )
    parser.add_argument("--ms2c_dp", type=str, default="data/ms2c/pubchem")
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--qos", type=str, default="m2")
    parser.add_argument("--gpu_type", type=str, default="rtx6000", choices=["rtx6000", "a40"])
    args = parser.parse_args()

    main(args)
