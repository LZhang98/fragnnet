import argparse
import os
import re
import subprocess

from fragnnet.runner import resolve_template_path


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


def _resolve_wandb_api_key(explicit_key: str | None = None) -> str | None:
    """Resolve W&B API key from multiple sources in priority order.

    Args:
        explicit_key: Key passed directly via CLI argument.

    Returns:
        API key string, or None if not found in any source.
    """
    if explicit_key:
        return explicit_key
    local_key_file = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wandb_api_key")
    )
    if os.path.isfile(local_key_file):
        with open(local_key_file) as f:
            key = f.read().strip()
        if key:
            return key
    import netrc as _netrc

    try:
        auth = _netrc.netrc().authenticators("api.wandb.ai")
        if auth:
            return auth[2]
    except Exception:
        pass
    return os.environ.get("WANDB_API_KEY")


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


_SLURM_TIME_RE = re.compile(r"^(\d+-)?(\d+)(:\d{2})?(:\d{2})?$")


def validate_slurm_time(time_str: str) -> None:
    """Validate SLURM --time format: [D-]HH[:MM[:SS]], MM, or MM:SS.

    Raises:
        ValueError: If the format is invalid.
    """
    if not _SLURM_TIME_RE.match(time_str):
        raise ValueError(
            f"Invalid --time format: {time_str!r}. "
            "Expected SLURM format: HH:MM:SS, D-HH:MM:SS, MM:SS, or MM."
        )


def validate_array_job_config(args) -> None:
    """Validate array job configuration."""
    if getattr(args, "num_sweep_jobs", None) is not None:
        if args.num_sweep_jobs <= 0:
            raise ValueError("--num_sweep_jobs must be > 0")
        if args.num_sweep_jobs > 1000:
            raise ValueError("--num_sweep_jobs must be <= 1000")
    if getattr(args, "num_sweep_agents", None) is not None:
        if args.num_sweep_agents <= 0:
            raise ValueError("--num_sweep_agents must be > 0")
        if args.num_sweep_agents > 1000:
            raise ValueError("--num_sweep_agents must be <= 1000")
        if getattr(args, "num_sweep_jobs", None) is None:
            raise ValueError("--num_sweep_agents requires --num_sweep_jobs to be set")


def validate_cv_config(args) -> None:
    """Validate CV array job configuration."""
    is_cv_mode = getattr(args, "cv_config_dp", None) is not None
    if not is_cv_mode:
        return
    if getattr(args, "cv_end", None) is None:
        raise ValueError("--cv_end is required when --cv_config_dp is set")
    if args.cv_start > args.cv_end:
        raise ValueError(f"--cv_start ({args.cv_start}) must be <= --cv_end ({args.cv_end})")
    if getattr(args, "num_sweep_jobs", None) is not None:
        raise ValueError("--cv_config_dp and --num_sweep_jobs are mutually exclusive")
    if args.cv_end - args.cv_start >= 100:
        raise ValueError(f"CV range too large (max 99 folds): {args.cv_start}-{args.cv_end}")


def main(args):
    if getattr(args, "list_gpu_profiles", False):
        print_gpu_profiles()
        return

    apply_gpu_profile(args)
    validate_slurm_time(args.time)
    validate_array_job_config(args)
    validate_cv_config(args)
    is_cv_mode = getattr(args, "cv_config_dp", None) is not None
    cv_config_dp_rel = args.cv_config_dp.lstrip("./").rstrip("/") if is_cv_mode else None

    if is_cv_mode and getattr(args, "cv_use_job_id", False):
        job_id_dp = os.path.join(os.path.abspath(args.frag_gnn_dp), "job_id")
        existing = [
            fold
            for fold in range(args.cv_start, args.cv_end + 1)
            if os.path.isfile(os.path.join(job_id_dp, f"{args.job_name}_cv{fold}.id"))
        ]
        print(
            f"> CV job_id check: {len(existing)}/{args.cv_end - args.cv_start + 1} folds have existing .id files"
        )
        if existing:
            print(f"  completed folds: {existing}")

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
            "#SBATCH --ntasks-per-node=1",
            f"#SBATCH --job-name={args.job_name}",
            f"#SBATCH --mem={args.mem}G",
            f"#SBATCH --cpus-per-task={args.cpu}",
            f"#SBATCH --time={args.time}",
            f"#SBATCH --mail-user={args.cc_email}",
            "#SBATCH --mail-type=ALL",
        ]
    )
    # add array job directive if num_sweep_jobs is specified
    if getattr(args, "num_sweep_jobs", None) is not None:
        num_jobs = args.num_sweep_jobs
        array_spec = f"0-{num_jobs - 1}"
        if getattr(args, "num_sweep_agents", None) is not None:
            array_spec += f"%{args.num_sweep_agents}"
        script_lines.append(f"#SBATCH --array={array_spec}")
    elif is_cv_mode:
        script_lines.append(f"#SBATCH --array={args.cv_start}-{args.cv_end}")

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
    if getattr(args, "num_sweep_jobs", None) is not None or is_cv_mode:
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
    if getattr(args, "num_sweep_jobs", None) is not None:
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
                'echo "Running sweep job ${AGENT_ID} of ${SLURM_ARRAY_TASK_MAX} total jobs (job ${SLURM_JOB_ID})"',
                "export AGENT_ID",
            ]
        )
    elif is_cv_mode:
        script_lines.extend(
            [
                "",
                "# CV fold array job setup",
                "CV_FOLD=${SLURM_ARRAY_TASK_ID}",
                'echo "Running CV fold ${CV_FOLD} (job ${SLURM_JOB_ID})"',
            ]
        )

    # load modules
    script_lines.extend(
        [
            "#load modules",
            "module load StdEnv/2023",
            "module load cuda/12.2",  # this is needed for torch_linear_assignment
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
            "    h5py\\",
            "    pyopenms==3.3.0 \\",
            "",
            "# intall from whl",
            "cp ${FRAGNNET_DP}/wheels/pyteomics-4.7.5-py3-none-any.whl ${WORKING_DIR}/pyteomics-4.7.5-py3-none-any.whl",
            "pip install pyteomics-4.7.5-py3-none-any.whl",
            "",
            "# install torch_linear_assignmnettorch_linear_assignment-0.0.6.tar.gz",
            "cp ${FRAGNNET_DP}/wheels/torch_linear_assignment-0.0.6.tar.gz ${WORKING_DIR}/torch_linear_assignment-0.0.6.tar.gz",
            "pip install --no-build-isolation torch_linear_assignment-0.0.6.tar.gz",
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
            "[ -f ${FRAGNNET_DP}/setup.cfg ] && cp ${FRAGNNET_DP}/setup.cfg ${WORKING_DIR}/",
            "[ -f ${FRAGNNET_DP}/setup.py ] && cp ${FRAGNNET_DP}/setup.py ${WORKING_DIR}/",
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
    config_template_fp = os.path.join(args.frag_gnn_dp, "config")
    config_d = {}

    # Check if running sweep script
    is_sweep_script = "run_wandb_sweep.py" in args.python_script_fp
    is_inference_eval_script = "run_save_inference_eval.py" in args.python_script_fp
    is_channel_selection_audit = (
        "mass_shatter_ion_h_top_k_sweep.py" in args.python_script_fp
    )

    # parse script kwargs
    script_kwargs_list = args.script_kwargs if args.script_kwargs else []
    for idx, kwarg in enumerate(script_kwargs_list):
        assert "=" in kwarg, kwarg
        key, value = kwarg.split("=", 1)
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

    # For inference eval scripts, inject --save_dp pointing inside WORKING_DIR unless already set
    if is_inference_eval_script:
        has_save_dp = any(kw.split("=")[0].lstrip("-") == "save_dp" for kw in script_kwargs_list)
        if not has_save_dp:
            python_line += " \\\n --save_dp output/inference_eval"

    # CV mode: use reference fold config for data discovery; inject dynamic config into python cmd
    if is_cv_mode:
        has_custom_config_in_kwargs = any(
            kw.split("=")[0].lstrip("-") in ("c", "custom_fp") for kw in script_kwargs_list
        )
        if has_custom_config_in_kwargs:
            raise ValueError(
                "Cannot use -c/--custom_fp in script_kwargs with --cv_config_dp. "
                "The CV config is selected automatically per array task."
            )
        if custom_config_fp is None:
            custom_config_fp = f"{cv_config_dp_rel}/cv{args.cv_start}.yml"
            if not os.path.isfile(custom_config_fp):
                raise FileNotFoundError(f"Reference CV config not found: {custom_config_fp}")
        python_line += f" \\\n -c {cv_config_dp_rel}/cv${{CV_FOLD}}.yml"

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

    # Templates are per model, so a directory has to be narrowed to one file before it can be
    # read for data paths. The custom config names the model; without one, fall back to FraGNNet.
    if os.path.isdir(config_template_fp):
        model_type = "frag_gnn"
        if custom_config_fp is not None:
            with open(custom_config_fp) as custom_file:
                for line in custom_file:
                    if line.startswith("model_type:"):
                        model_type = line.split(" #")[0].strip().split(":")[-1].strip()
                        break
        config_template_fp = resolve_template_path(config_template_fp, model_type)
        config_d["config_template"] = config_template_fp

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
    if config_d.get("model_ckpt") and not os.path.isabs(config_d["model_ckpt"]):
        data_keyword.append("model_ckpt")
    # copy fp if pretraining

    if config_d["finetune"].lower() == "true":
        data_keyword.append("finetune_pretrained_ckpt")

    if is_sweep_script:
        print(f"> running wandb sweep script with model_type: {config_d['model_type']}")
    else:
        print(f"> model type: {config_d['model_type']}")

    # copy data files
    script_lines.extend(["#copy files"])

    frag_gnn_dp_abs = os.path.abspath(args.frag_gnn_dp).rstrip("/") + "/"
    for name in data_keyword:
        path_value = config_d[name]

        if path_value.startswith("./"):
            path_value = path_value[2:]
        # Strip frag_gnn_dp prefix so absolute paths don't double up in rsync --relative.
        if os.path.isabs(path_value) and path_value.startswith(frag_gnn_dp_abs):
            path_value = path_value[len(frag_gnn_dp_abs) :]
        if name not in config_d:
            raise ValueError(f"missing {name} in config file")

        if name == "frag_dp":
            # check if frag_dp is a file or dir, if it is file, copy it and unpack
            tar_ext = None
            print(path_value)
            if args.python_script_fp.rsplit("/", 1)[-1] in [
                "run_pl_model_fit.py",
                "run_wandb_sweep.py",
                "run_save_inference_eval.py",
                "node_vs_h_transfer_diagnostic.py",
                "debug_identity_shortcut_eval.py",
            ]:
                frag_dp = path_value.rsplit("/", 1)[0]  # strip /dags suffix to get parent dir
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
        elif is_cv_mode and name == "config":
            # CV mode: copy entire cv config directory instead of just the reference file
            print(f"> CV mode: copying cv config directory {cv_config_dp_rel}")
            script_lines.extend(
                [
                    f"# copy CV config directory {cv_config_dp_rel} to slurm tmp dir",
                    f"rsync -av --relative ${{FRAGNNET_DP}}/./{cv_config_dp_rel} ${{WORKING_DIR}}",
                ]
            )
        elif is_cv_mode and name == "split_dp":
            # CV mode: validate all fold split dirs exist, then emit one dynamic rsync
            print(f"> CV mode: copying split dirs for folds {args.cv_start}-{args.cv_end}")
            fold_split_dps: dict[int, str] = {}
            for fold_idx in range(args.cv_start, args.cv_end + 1):
                fold_config_fp = f"{cv_config_dp_rel}/cv{fold_idx}.yml"
                fold_split_dp = None
                with open(fold_config_fp) as _f:
                    for line in _f:
                        if line.startswith("split_dp:"):
                            fold_split_dp = line.split(":", 1)[1].split("#")[0].strip().lstrip("./")
                            break
                if fold_split_dp is None:
                    raise ValueError(f"split_dp not found in {fold_config_fp}")
                if not os.path.exists(fold_split_dp):
                    raise FileNotFoundError(f"Split dir not found: {fold_split_dp}")
                fold_split_dps[fold_idx] = fold_split_dp
            # detect consistent pattern by replacing cv{fold_idx} with cv${CV_FOLD}
            patterns = {
                fold_idx: path.replace(f"cv{fold_idx}", "cv${CV_FOLD}")
                for fold_idx, path in fold_split_dps.items()
            }
            unique_patterns = set(patterns.values())
            if len(unique_patterns) == 1:
                dynamic_path = unique_patterns.pop()
                print(f"> adding cmd to copy split dir {dynamic_path} (dynamic CV fold)")
                script_lines.extend(
                    [
                        f"# copy split_dp for CV fold ${{CV_FOLD}}: {dynamic_path}",
                        f"rsync -av --relative ${{FRAGNNET_DP}}/./{dynamic_path} ${{WORKING_DIR}}",
                    ]
                )
            else:
                # paths don't share a common cv{n} pattern — copy all unique dirs statically
                for fold_idx, fold_split_dp in fold_split_dps.items():
                    print(f"> adding cmd to copy split dir {fold_split_dp} (fold {fold_idx})")
                    script_lines.extend(
                        [
                            f"# copy split_dp for CV fold {fold_idx}: {fold_split_dp}",
                            f"rsync -av --relative ${{FRAGNNET_DP}}/./{fold_split_dp} ${{WORKING_DIR}}",
                        ]
                    )
        elif is_cv_mode and name == "finetune_pretrained_ckpt":
            # CV mode: validate all fold ckpts exist, then emit bash array + single rsync
            print(f"> CV mode: copying pretrained ckpts for folds {args.cv_start}-{args.cv_end}")
            fold_ckpt_fps: dict[int, str] = {}
            for fold_idx in range(args.cv_start, args.cv_end + 1):
                fold_config_fp = f"{cv_config_dp_rel}/cv{fold_idx}.yml"
                fold_ckpt_fp = None
                with open(fold_config_fp) as _f:
                    for line in _f:
                        if line.startswith("finetune_pretrained_ckpt:"):
                            fold_ckpt_fp = line.split(":", 1)[1].split("#")[0].strip().lstrip("./")
                            break
                if fold_ckpt_fp is None:
                    raise ValueError(f"finetune_pretrained_ckpt not found in {fold_config_fp}")
                if not os.path.exists(fold_ckpt_fp):
                    raise FileNotFoundError(f"Pretrained ckpt not found: {fold_ckpt_fp}")
                fold_ckpt_fps[fold_idx] = fold_ckpt_fp
            print("> adding bash array + single rsync for per-fold pretrained ckpts")
            array_lines = ["# copy pretrained ckpt for current CV fold (array-indexed by CV_FOLD)"]
            array_lines.append("declare -A CV_PRETRAINED_CKPT")
            for fold_idx, fold_ckpt_fp in fold_ckpt_fps.items():
                array_lines.append(f'CV_PRETRAINED_CKPT[{fold_idx}]="{fold_ckpt_fp}"')
            array_lines.append(
                "rsync -av --relative ${FRAGNNET_DP}/./${CV_PRETRAINED_CKPT[${CV_FOLD}]}"
                " ${WORKING_DIR}"
            )
            script_lines.extend(array_lines)
        elif name == "model_ckpt":
            print(f"> adding cmd to copy {name} file from {path_value} to slurm tmp dir")
            script_lines.extend(
                [
                    f"# copy {name} file from {path_value} to slurm tmp dir",
                    f"rsync -avL --relative ${{FRAGNNET_DP}}/./{path_value} ${{WORKING_DIR}}",
                    f"if [[ ! -f ${{WORKING_DIR}}/{path_value} ]]; then",
                    f'    echo "ERROR: checkpoint was not staged: ${{WORKING_DIR}}/{path_value}" >&2',
                    "    exit 1",
                    "fi",
                ]
            )
        elif name == "config_template":
            # `config/` is not staged wholesale, so the resolved template arrives alone. The
            # sibling templates have to come with it: `load_config` reads them to tell a key
            # owned by another model from a typo, and without them a historical config that
            # carries a foreign key fails validation on the node but passes locally.
            print(f"> adding cmd to copy {name} file from {path_value} to slurm tmp dir")
            sibling_glob = os.path.join(os.path.dirname(path_value), "*_template.yml")
            script_lines.extend(
                [
                    f"# copy {name} file from {path_value} to slurm tmp dir",
                    f"rsync -av --relative ${{FRAGNNET_DP}}/./{path_value} ${{WORKING_DIR}}",
                    "# copy the other model templates so config validation can see them",
                    f'if compgen -G "${{FRAGNNET_DP}}/{sibling_glob}" > /dev/null; then',
                    f"    rsync -av --relative ${{FRAGNNET_DP}}/./{sibling_glob} ${{WORKING_DIR}}",
                    "fi",
                ]
            )
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

    # resolve job_id: dynamic per-fold (cv_use_job_id) or static from script_kwargs
    cv_use_job_id = is_cv_mode and getattr(args, "cv_use_job_id", False)
    static_job_id = next(
        (
            kw.split("=", 1)[1]
            for kw in script_kwargs_list
            if kw.split("=")[0].lstrip("-") in ("j", "job_id")
        ),
        None,
    )
    job_id_name = f"{args.job_name}_cv${{CV_FOLD}}" if cv_use_job_id else static_job_id

    if job_id_name:
        script_lines.extend(
            [
                "# create job_id directory if not exists",
                "if [ ! -d ${FRAGNNET_DP}/job_id ]; then",
                "    mkdir ${FRAGNNET_DP}/job_id",
                "fi",
                "# symlink job_id folder for CV fold resume",
                "if [ ! -e job_id ]; then",
                "    ln -s ${FRAGNNET_DP}/job_id job_id",
                "fi",
            ]
        )

    # for sweep scripts, append SLURM job/agent ID so wandb run names are unique
    # job name is used for local sweep, since fir and vulcan now support wandb agent directly,
    # job id is used for narval only, and it maybe broken.
    if is_sweep_script:
        if getattr(args, "num_sweep_jobs", None) is not None:
            python_line += " \\\n -p ${SLURM_JOB_ID}_${AGENT_ID}"
        else:
            python_line += " \\\n -p ${SLURM_JOB_ID}"
    elif cv_use_job_id:
        python_line += f" \\\n -j {args.job_name}_cv${{CV_FOLD}}"

    wandb_api_key = _resolve_wandb_api_key(getattr(args, "wandb_api_key", None))
    if wandb_api_key:
        script_lines.extend(["# set wandb api key", f"export WANDB_API_KEY={wandb_api_key}", ""])

    # wait for GPU to be available (MIG instances can be briefly busy at job start)
    script_lines.extend(
        [
            "# wait for GPU to be available (MIG instances can be briefly busy at job start)",
            "MAX_WAIT=120",
            "WAITED=0",
            "while ! python -c \"import torch; torch.zeros(1, device='cuda')\" 2>/dev/null; do",
            "    if [ ${WAITED} -ge ${MAX_WAIT} ]; then",
            '        echo "ERROR: GPU not available after ${MAX_WAIT}s, giving up"',
            "        exit 1",
            "    fi",
            '    echo "GPU not yet available, retrying in 10s... (${WAITED}s elapsed)"',
            "    sleep 10",
            "    WAITED=$((WAITED + 10))",
            "done",
            'echo "GPU is available, starting training"',
            "",
        ]
    )

    # add python line
    script_lines.extend(["# run python script", "cd ${WORKING_DIR}", python_line])

    # Preserve the channel-selection audit CSV outside node-local scratch.
    if is_channel_selection_audit:
        script_lines.append("CHANNEL_AUDIT_STATUS=$?")
        output_fp = config_d.get(
            "output_fp",
            "output/eval/mass_shatter_ion_h_top_k_sweep.csv",
        )
        if not os.path.isabs(output_fp):
            output_src = os.path.join("${WORKING_DIR}", output_fp)
            output_dest = os.path.abspath(os.path.join(args.frag_gnn_dp, output_fp))
            script_lines.extend(
                [
                    "",
                    "# copy channel-selection audit output back to the project directory",
                    f"mkdir -p {os.path.dirname(output_dest)}",
                    f"if [[ -f {output_src} ]]; then",
                    f"    rsync -av {output_src} {output_dest}",
                    "fi",
                ]
            )
        script_lines.extend(
            [
                'if [[ "${CHANNEL_AUDIT_STATUS}" -ne 0 ]]; then',
                '    exit "${CHANNEL_AUDIT_STATUS}"',
                "fi",
            ]
        )

    # copy predictions output back to project directory
    if is_inference_eval_script:
        dest_dp = os.path.abspath(args.inference_save_dp)
        script_lines.extend(
            [
                "",
                "# copy inference eval predictions back to local directory",
                f"mkdir -p {dest_dp}",
                f"rsync -av ${{WORKING_DIR}}/output/inference_eval/ {dest_dp}/",
            ]
        )

    script_lines.extend(
        ["", "# record ending", "echo `date`: Job ${SLURM_JOB_ID} finished running"]
    )

    script_str = "\n".join(script_lines)
    if not script_str.startswith("#!"):
        raise ValueError("Generated script must start with a shebang (#!).")

    # Defensive normalization to avoid accidental leading blank lines/BOM.
    script_str = script_str.lstrip("\ufeff")
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
        "--inference_save_dp",
        type=str,
        default="output/inference_eval",
        help="Local directory to rsync inference eval predictions into after the job completes",
    )
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
    parser.add_argument(
        "--time",
        type=str,
        default="3:00:00",
        help="Job wall-clock time limit in SLURM format: D-HH:MM:SS, HH:MM:SS, or H (hours)",
    )
    parser.add_argument("--script_kwargs", type=str, nargs="+", required=False)
    parser.add_argument("--job_name", type=str, required=False, help="name of the job")
    parser.add_argument(
        "--num_sweep_jobs",
        type=int,
        default=None,
        help=(
            "Total number of array jobs for wandb sweep (creates SLURM array job). "
            "Example: --num_sweep_jobs=20 creates --array=0-19"
        ),
    )
    parser.add_argument(
        "--num_sweep_agents",
        type=int,
        default=None,
        help=(
            "Max number of concurrently running sweep jobs (throttle). "
            "Requires --num_sweep_jobs. "
            "Example: --num_sweep_jobs=20 --num_sweep_agents=4 creates --array=0-19%%4"
        ),
    )
    parser.add_argument(
        "--cv_config_dp",
        type=str,
        default=None,
        help=(
            "Directory containing cv{n}.yml config files for cross-validation array jobs. "
            "Creates a SLURM array job with one task per CV fold (--cv_start to --cv_end). "
            "Example: --cv_config_dp config/spectraverse/neims --cv_start 0 --cv_end 9"
        ),
    )
    parser.add_argument(
        "--cv_start",
        type=int,
        default=0,
        help="First CV fold index (inclusive). Used with --cv_config_dp.",
    )
    parser.add_argument(
        "--cv_end",
        type=int,
        default=None,
        help="Last CV fold index (inclusive). Required with --cv_config_dp.",
    )
    parser.add_argument(
        "--cv_use_job_id",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Pass -j {job_name}_cv${CV_FOLD} to the python script for per-fold checkpoint IDs.",
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
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help=(
            "W&B API key to export in the job script. "
            "Falls back to ~/.wandb_api_key (plain text file) then WANDB_API_KEY env var."
        ),
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
