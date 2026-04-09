import argparse
import os
import subprocess


def _normalize_profile_name(name: str) -> str:
    return name.strip().lower().replace(" ", "")


GPU_PROFILES = {
    # Fir (H100)
    "fir:h100-80gb": {
        "cluster": "fir",
        "gpu_label": "H100-80gb",
        "gres_gpu": "gpu:h100:1",
        "cpu": 12,
        "mem": 280,
        "price": 12.2,
        "notes": "12 cores, 288GB (ideal); 12 cores, 280GB (sbatch)",
    },
    "fir:h100-1g.10gb": {
        "cluster": "fir",
        "gpu_label": "H100-1g.10gb",
        "gres_gpu": "gpu:h100_1g.10gb:1",
        "cpu": 1,
        "mem": 35,
        "price": 1.74,
        "notes": "1.7 cores, 41GB (ideal); 1 core, 35GB (sbatch)",
    },
    "fir:h100-2g.20gb": {
        "cluster": "fir",
        "gpu_label": "H100-2g.20gb",
        "gres_gpu": "gpu:h100_2g.20gb:1",
        "cpu": 3,
        "mem": 70,
        "price": 3.48,
        "notes": "3.4 cores, 82GB (ideal); 3 cores, 70GB (sbatch)",
    },
    "fir:h100-3g.40gb": {
        "cluster": "fir",
        "gpu_label": "H100-3g.40gb",
        "gres_gpu": "gpu:h100_3g.40gb:1",
        "cpu": 6,
        "mem": 140,
        "price": 6.1,
        "notes": "6 cores, 144GB (ideal); 6 cores, 140GB (sbatch)",
    },
    # Narval (A100)
    "narval:a100-40gb": {
        "cluster": "narval",
        "gpu_label": "A100-40gb",
        "gres_gpu": "gpu:a100:1",
        "cpu": 12,
        "mem": 124,
        "price": 4.0,
        "notes": "12 cores, 124.5GB (ideal); 12 cores, 124GB (sbatch)",
    },
    "narval:a100-1g.5gb": {
        "cluster": "narval",
        "gpu_label": "A100-1g.5gb",
        "gres_gpu": "gpu:a100_1g.5gb:1",
        "cpu": 1,
        "mem": 15,
        "price": 0.57,
        "notes": "1.7 cores, 17.7GB (ideal); 1 core, 15GB (sbatch)",
    },
    "narval:a100-2g.10gb": {
        "cluster": "narval",
        "gpu_label": "A100-2g.10gb",
        "gres_gpu": "gpu:a100_2g.10gb:1",
        "cpu": 3,
        "mem": 31,
        "price": 1.14,
        "notes": "3.4 cores, 35.4GB (ideal); 3 cores, 31GB (sbatch)",
    },
    "narval:a100-3g.20gb": {
        "cluster": "narval",
        "gpu_label": "A100-3g.20gb",
        "gres_gpu": "gpu:a100_3g.20gb:1",
        "cpu": 6,
        "mem": 62,
        "price": 2.0,
        "notes": "6.0 cores, 62.2GB (ideal); 6 cores, 62GB (sbatch)",
    },
    "narval:a100-4g.20gb": {
        "cluster": "narval",
        "gpu_label": "A100-4g.20gb",
        "gres_gpu": "gpu:a100_4g.20gb:1",
        "cpu": 6,
        "mem": 62,
        "price": 2.3,
        "notes": "6.9 cores, 71.5GB (ideal); 6 cores, 62GB (sbatch)",
    },
    "vulcan:l40s": {
        "cluster": "vulcan",
        "gpu_label": "l40s",
        "gres_gpu": "gpu:1",
        "cpu": 6,
        "mem": 192,
        "price": 0,
        "notes": "vulcan is free to use for all aip users",
    },
}


GPU_PROFILE_ALIASES = {
    # allow selecting by GPU label alone when unambiguous
    "h100-80gb": "fir:h100-80gb",
    "h100-1g.10gb": "fir:h100-1g.10gb",
    "h100-2g.20gb": "fir:h100-2g.20gb",
    "h100-3g.40gb": "fir:h100-3g.40gb",
    "a100-40gb": "narval:a100-40gb",
    "a100-1g.5gb": "narval:a100-1g.5gb",
    "a100-2g.10gb": "narval:a100-2g.10gb",
    "a100-3g.20gb": "narval:a100-3g.20gb",
    "a100-4g.20gb": "narval:a100-4g.20gb",
    "a6000": "vulcan:l40s",
}


def print_gpu_profiles() -> None:
    print("Available --gpu_profile presets:")
    for key in sorted(GPU_PROFILES.keys()):
        p = GPU_PROFILES[key]
        print(
            f"- {key}\t({p['gpu_label']}, cpu={p['cpu']}, mem={p['mem']}G, gres={p['gres_gpu']}, price={p['price']})"
        )


def apply_gpu_profile(args) -> None:
    if not args.gpu_profile:
        return

    name = _normalize_profile_name(args.gpu_profile)
    name = GPU_PROFILE_ALIASES.get(name, name)
    if name not in GPU_PROFILES:
        raise ValueError(
            f"Unknown gpu profile: {args.gpu_profile}. Use --list_gpu_profiles to see valid names."
        )

    p = GPU_PROFILES[name]
    # override resource requests
    args.cpu = int(p["cpu"])
    args.mem = int(p["mem"])
    # prefer --gres for MIG/full GPU selection
    args.gres_gpu = p["gres_gpu"]
    # ensure we still request a GPU when the old --gpu path is used elsewhere
    if getattr(args, "gpu", 0) == 0:
        args.gpu = 1


def validate_array_job_config(args) -> None:
    """Validate array job configuration."""
    if getattr(args, "num_sweep_agents", None) is not None:
        if args.num_sweep_agents <= 0:
            raise ValueError("--num_sweep_agents must be > 0")
        if args.num_sweep_agents > 1000:
            raise ValueError("--num_sweep_agents must be <= 1000")


def main(args):
    if getattr(args, "list_gpu_profiles", False):
        print_gpu_profiles()
        return

    apply_gpu_profile(args)
    validate_array_job_config(args)

    # create folders
    os.makedirs(f"{os.path.abspath(args.frag_gnn_dp)}/wandb", exist_ok=True)
    os.makedirs(f"{os.path.abspath(args.script_dp)}", exist_ok=True)
    os.makedirs(f"{os.path.abspath(args.job_dp)}", exist_ok=True)

    script_lines = []
    # account, ram, cpu, gpu, and etc
    script_lines.extend(
        [
            "#!/bin/bash",
            f"#SBATCH --account={args.cc_account}",
            "#SBATCH --nodes=1",
            f"#SBATCH --job-name={args.job_name}",
            f"#SBATCH --mem={args.mem}G",
            f"#SBATCH --cpus-per-task={args.cpu}",
            f"#SBATCH --time=0-{args.time}:00:00",
            f"#SBATCH --mail-user={args.cc_email}",
            "#SBATCH --mail-type=ALL",
        ]
    )
    # add array job directive if num_sweep_agents is specified
    if getattr(args, "num_sweep_agents", None) is not None:
        num_agents = args.num_sweep_agents
        # SLURM array indices are 0-based
        script_lines.append(f"#SBATCH --array=0-{num_agents - 1}")

    # add gpus
    if getattr(args, "gres_gpu", None):
        script_lines.append(f"#SBATCH --gres={args.gres_gpu}")
    elif getattr(args, "gpus_per_node", None):
        script_lines.append(f"#SBATCH --gpus-per-node={args.gpus_per_node}")
    elif args.gpu > 0:
        # eg. --gpus-per-node=v100:1 or --gpus-per-node=1
        gpu_type = getattr(args, "gpu_type", None)
        if gpu_type is not None:
            script_lines.append(f"#SBATCH --gpus-per-node={gpu_type}:{args.gpu}")
        else:
            script_lines.append(f"#SBATCH --gpus-per-node={args.gpu}")

    # config where to save job output and job error
    # append to existing onces
    # For array jobs, include task ID in output filename
    if getattr(args, "num_sweep_agents", None) is not None:
        job_output = f"{os.path.abspath(args.job_dp)}/{args.job_name}_%j_%a.out"
        job_error = f"{os.path.abspath(args.job_dp)}/{args.job_name}_%j_%a.err"
    else:
        job_output = f"{os.path.abspath(args.job_dp)}/{args.job_name}_%j.out"
        job_error = f"{os.path.abspath(args.job_dp)}/{args.job_name}_%j.err"

    script_lines.extend(
        [
            f"#SBATCH --output={job_output}",
            f"#SBATCH --error={job_error}",
            "#SBATCH --open-mode=append",
        ]
    )
    # start recording
    script_lines.extend(
        ["", "# record start", "echo `date`: Job ${SLURM_JOB_ID} is allocated resources"]
    )

    # handle array jobs for wandb sweep
    if getattr(args, "num_sweep_agents", None) is not None:
        script_lines.extend(
            [
                "",
                "# array job setup for wandb sweep",
                'if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then',
                "    AGENT_ID=0",
                "else",
                "    AGENT_ID=${SLURM_ARRAY_TASK_ID}",
                "fi",
                'echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"',
                'echo "Running as sweep agent ${AGENT_ID} of ${SLURM_ARRAY_TASK_MAX} agents (job ${SLURM_JOB_ID})"',
                "export AGENT_ID",
            ]
        )

    # load modules
    script_lines.extend(
        [
            "#load modules",
            "module load StdEnv/2023",
            f"module load python/{args.python_version}",
            f"module load rdkit/{args.rdkit_version}",
        ]
    )

    # create venv on compute node
    script_lines.extend(
        [
            "",
            # "# go to script dir",
            # f"cd {os.path.abspath(args.frag_gnn_dp)}",
            f"FRAGNNET_DP={os.path.abspath(args.frag_gnn_dp)}",
            "WORKING_DIR=${SLURM_TMPDIR}/${SLURM_JOB_ID}",
            "# create work dir",
            "mkdir -p ${WORKING_DIR}",
            "# create local env",
            # go to slurm and create virtual env
            "cd  ${WORKING_DIR}",
            "virtualenv FRAG-GNN",
            "source FRAG-GNN/bin/activate",
            "pip install --no-index --upgrade pip",
            "pip install --no-index torch==2.4.1 \\",
            "    torch_geometric==2.6.1 \\",
            "    torch_scatter==2.1.2 \\",
            "    pandas \\",
            "    wandb==0.21.2 \\",  # 0.24 is known to have upload bugs
            "    tqdm \\",
            "    scipy==1.15.2 \\",
            "    joblib==1.3.2 \\",
            "    matplotlib \\",
            "    lightning==2.4.0 \\",
            "    dgl==2.4.0 \\",
            "    omegaconf \\",
            "    cython==3.0.12 \\",
            "    pandarallel==1.6.5 \\",
            "    lmdb==1.7.5 \\",
            "    h5py",
            "",
            "# intall from whl",
            "cp ${FRAGNNET_DP}/wheels/pyteomics-4.7.5-py3-none-any.whl ${WORKING_DIR}/pyteomics-4.7.5-py3-none-any.whl",
            "pip install pyteomics-4.7.5-py3-none-any.whl",
            "",
        ]
    )

    # copy fragnnet sourfce code and install to venv
    script_lines.extend(
        [
            "# copy script code",
            'rsync -av --exclude=".*" --exclude="__pycache__" ${FRAGNNET_DP}/preproc_scripts ${WORKING_DIR}',
            'rsync -av --exclude=".*" --exclude="__pycache__" ${FRAGNNET_DP}/scripts ${WORKING_DIR}',
            "# copy src code - this is needed to avoid issues with multiple install read the same src code",
            'rsync -av --exclude=".*" --exclude="__pycache__" --exclude="*.so" ${FRAGNNET_DP}/src ${WORKING_DIR}',
            "cp ${FRAGNNET_DP}/setup.cfg ${WORKING_DIR}/",
            "cp ${FRAGNNET_DP}/setup.py ${WORKING_DIR}/",
            "cp ${FRAGNNET_DP}/pyproject.toml ${WORKING_DIR}/",
            "# install with out dependencies, assume all dependencies are installed manually",
            "pip install -I --no-deps ${WORKING_DIR}",
            "",
        ]
    )

    # https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html
    # lightning seems have issues with srun
    # disable DDP in the trainer
    python_line = f"python {args.python_script_fp}"

    custom_config_fp = None
    # default config template
    config_template_fp = os.path.join(args.frag_gnn_dp, "config", "template.yml")
    config_d = {}

    # Check if running sweep script
    is_sweep_script = "run_wandb_sweep.py" in args.python_script_fp

    # parse script kwargs
    script_kwargs_list = args.script_kwargs if args.script_kwargs else []
    for idx, kwarg in enumerate(script_kwargs_list):
        assert "=" in kwarg, kwarg
        key, value = kwarg.split("=")
        # this useful for inference setup where some config are overridden via command line
        config_d[key] = value
        if idx == 0:
            python_line += " \\\n"  # first arg
        if len(key) == 1:
            python_line += f" -{key} {value}"
        elif key.startswith("-"):
            python_line += f" {key} {value}"
        else:
            python_line += f" --{key} {value}"
        # add line breaks
        if idx != len(script_kwargs_list) - 1:
            python_line += " \\\n"

        if key == "c" or key == "custom_fp":
            custom_config_fp = value
        if key == "t" or key == "template_fp":
            config_template_fp = value

    # For sweep scripts with array jobs, append --job_id
    if is_sweep_script and getattr(args, "num_sweep_agents", None) is not None:
        if len(script_kwargs_list) > 0:
            python_line += " \\\n"
        python_line += " -j ${AGENT_ID}"

    # parse configs    file
    config_keywords = [
        "spec_fp",
        "mol_fp",
        "ann_fp",
        "mces_fp",
        "config",
        "config_template",
        "split_dp",
        "magma_dp",
        "frag_dp",
        "model_type",
        "finetune",
        "finetune_pretrained_ckpt",
    ]
    config_d["config_template"] = config_template_fp
    if custom_config_fp is not None:
        config_d["config"] = custom_config_fp

    # parse configs
    # read from template first, then custom config,
    for config_fp in [custom_config_fp, config_template_fp]:
        if config_fp is not None:
            with open(config_fp) as config_file:
                config_lines = config_file.readlines()
                for line in config_lines:
                    for config_keyword in config_keywords:
                        # this useful for inference setup where some paths are overridden via command line
                        if line.startswith(config_keyword + ":") and config_keyword not in config_d:
                            config_d[config_keyword] = (
                                line.split(" #")[0].strip().split(":")[-1].strip()
                            )

    # change keywords based model type
    data_keyword = ["config", "config_template", "spec_fp", "mol_fp", "split_dp"]

    if config_d["model_type"] in ["frag_gnn", "fragnnet"]:
        data_keyword.append("frag_dp")
    elif config_d["model_type"] in ["iceberg_gen", "iceberg_inten"]:
        data_keyword.append("magma_dp")
    elif config_d["model_type"] == "graff":
        data_keyword.append("ann_fp")
    elif config_d["model_type"] == "mces_pretraining":
        data_keyword.append("mces_fp")
    # copy fp if pretraining

    if config_d["finetune"] == "true":
        data_keyword.append(config_d["finetune_pretrained_ckpt"])

    if is_sweep_script:
        print(f"> running wandb sweep script with model_type: {config_d['model_type']}")
    else:
        print(f"> model type: {config_d['model_type']}")

    # copy data files
    script_lines.extend(["#copy files"])

    for name in data_keyword:
        path_value = config_d[name]

        if path_value.startswith("./"):
            path_value = path_value[2:]
        if name not in config_d:
            raise ValueError(f"missing {name} in config file")

        if name == "frag_dp":
            # check if frag_dp is a file or dir, if it is file, copy it and unpack
            tar_ext = None
            print(path_value)
            if args.python_script_fp.rsplit("/", 1)[-1] in [
                "run_pl_model_fit.py",
                "run_wandb_sweep.py",
            ]:
                frag_dp = path_value.rsplit("/", 1)[0]  # this is for training setup
            else:
                frag_dp = path_value
            # use tarball and then unpacks
            for ext in [".tar", ".tar.xz", ".tar.gz", ".tgz"]:
                if os.path.isfile(frag_dp + ext):
                    tar_ext = ext
                    break
            if tar_ext is not None:
                print(f"> adding cmd to copy {frag_dp + tar_ext}")
                script_lines.extend(
                    [
                        f"# copy {name} file from {frag_dp}{tar_ext} to slurm tmp dir and unpack",
                        f"rsync -av --relative ${{FRAGNNET_DP}}/./{frag_dp + tar_ext} ${{WORKING_DIR}}",
                        f"tar -xf ${{WORKING_DIR}}/{frag_dp + tar_ext} -C ${{WORKING_DIR}}/{frag_dp.rsplit('/', 1)[0]}",
                    ]
                )
            elif os.path.isdir(frag_dp):
                print(f"> adding cmd to copy directory {frag_dp}")
                script_lines.extend(
                    [
                        f"# copy {name} file from {frag_dp} to slurm tmp dir",
                        f"rsync -av --relative ${{FRAGNNET_DP}}/./{frag_dp} ${{WORKING_DIR}}",
                    ]
                )
            else:
                raise FileNotFoundError(f"File not found: {frag_dp} or {frag_dp + ext}")
        elif not os.path.exists(path_value):
            raise FileNotFoundError(f"File not found: {path_value}")
        else:
            print(f"> adding cmd to copy {name} file from {path_value} to slurm tmp dir")
            script_lines.extend(
                [
                    f"# copy {name} file from {path_value} to slurm tmp dir",
                    f"rsync -av --relative ${{FRAGNNET_DP}}/./{path_value} ${{WORKING_DIR}}",
                ]
            )
    # soft link back to home directy for wandb
    script_lines.extend(
        [
            "# create wandb directory if not exists",
            "if [ ! -d ${FRAGNNET_DP}/wandb ]; then",
            "    mkdir ${FRAGNNET_DP}/wandb",
            "fi",
            "# soft link to wandb",
            "cd ${WORKING_DIR}",
            "if [ ! -e wandb ]; then",
            "    ln -s ${FRAGNNET_DP}/wandb wandb",
            "fi",
        ]
    )

    # add python line
    script_lines.extend(["# run python script", "cd ${WORKING_DIR}", python_line])

    script_lines.extend(
        ["", "# record ending", "echo `date`: Job ${SLURM_JOB_ID} finished running"]
    )

    script_str = "\n".join(script_lines)
    assert os.path.isdir(args.script_dp), args.script_dp
    script_fp = os.path.join(args.script_dp, f"{args.job_name}.sh")
    with open(script_fp, "w") as script_file:
        print(script_str, file=script_file)

    print("=" * 80)
    if not args.no_submit:
        print(f"submitting job script {script_fp} to slurm")
        process = subprocess.Popen(
            ["sbatch", script_fp],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        out, err = process.communicate()
        print(out.strip(), err.strip())
    else:
        print("Slrum script created. Not submitted due to --no_submit=True\n")
        print("To submit, run:")
        print(f"sbatch {script_fp} \n")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script_dp", type=str, default="slurm_scripts")
    parser.add_argument("--job_dp", type=str, default="jobs")
    parser.add_argument(
        "--python_script_fp", type=str, required=True, help="python script to run on cc"
    )
    parser.add_argument(
        "--frag_gnn_dp", type=str, help="location of frag-gnn project", required=True
    )
    parser.add_argument("--mem", type=int, default=128)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=1)

    parser.add_argument(
        "--gpu_profile",
        type=str,
        default=None,
        help=(
            "Apply a preset resource profile (overrides --cpu/--mem and sets --gres). "
            "Examples: fir:h100-1g.10gb, narval:a100-40gb. Use --list_gpu_profiles to view all."
        ),
    )
    parser.add_argument(
        "--list_gpu_profiles",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Print available --gpu_profile presets and exit",
    )
    parser.add_argument(
        "--gpus_per_node",
        type=str,
        default=None,
        help="Raw value for #SBATCH --gpus-per-node (overrides --gpu/--gpu_type when set)",
    )
    parser.add_argument(
        "--gres_gpu",
        type=str,
        default=None,
        help="Raw value for #SBATCH --gres (e.g. gpu:a100:1 or gpu:h100_1g.10gb:1)",
    )
    parser.add_argument("--time", type=int, default=3)
    parser.add_argument("--script_kwargs", type=str, nargs="+", required=False)
    parser.add_argument("--job_name", type=str, required=False, help="name of the job")
    parser.add_argument(
        "--num_sweep_agents",
        type=int,
        default=None,
        help=(
            "Number of parallel agents for wandb sweep (creates SLURM array job). "
            "Each agent will run the same sweep configuration independently. "
            "Example: --num_sweep_agents=10 creates --array=0-9"
        ),
    )
    parser.add_argument(
        "--cc_account",
        default="rrg-dwishart",
        choices=["rrg-dwishart", "rrg-rgreiner", "def-dwishart", "def-rgreiner", "aip-rgreiner"],
    )
    parser.add_argument("--cc_email", help="email", default="fw4@ualberta.ca")
    parser.add_argument(
        "--python_version", default="3.11", choices=["3.10", "3.11"], help="python version"
    )
    parser.add_argument(
        "--rdkit_version",
        default="2023.09.5",
        choices=["2022.03.5", "2023.09.5", "2024.03.5"],
        help="rdkit version",
    )
    parser.add_argument("--no_submit", type=bool, default=True)
    args = parser.parse_args()
    # if not getattr(args, "list_gpu_profiles", False):
    #    missing = []
    #    for required_arg in ["python_script_fp", "frag_gnn_dp", "job_name"]:
    #        if getattr(args, required_arg, None) in [None, ""]:
    #            missing.append(required_arg)
    #    if missing:
    #        parser.error(f"Missing required arguments: {', '.join('--' + m for m in missing)}")

    main(args)
