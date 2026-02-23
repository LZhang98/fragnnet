import importlib.util
from argparse import Namespace
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    fp = repo_root / "scripts" / "setup_job_cc.py"
    spec = importlib.util.spec_from_file_location("setup_job_cc", str(fp))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sanitize_job_name():
    mod = load_module()
    assert mod._sanitize_job_name("/path/to/my-config.yaml") == "my-config"
    assert mod._sanitize_job_name("weird name@v1.yml") == "weird-name-v1"
    assert mod._sanitize_job_name("") == "job"


def test_build_python_line_and_override():
    mod = load_module()
    kwargs = ["config=cfg.yaml", "a=1", "verbose=true"]
    line = mod._build_python_line(kwargs, "scripts/run.py")
    assert "python scripts/run.py" in line
    assert "--config cfg.yaml" in line
    # override config
    line2 = mod._build_python_line(kwargs, "scripts/run.py", override_config="other.yaml")
    assert "--config other.yaml" in line2


def test_config_list_creates_scripts(tmp_path):
    mod = load_module()
    args = Namespace()
    args.list_gpu_profiles = False
    args.gpu_profile = None
    args.frag_gnn_dp = str(tmp_path)
    args.script_dp = str(tmp_path / "scripts_out")
    args.job_dp = str(tmp_path / "jobs_out")
    args.python_script_fp = "scripts/run_pl_model_fit.py"
    args.mem = 4
    args.cpu = 1
    args.gpu = 0
    args.gpus_per_node = None
    args.gres_gpu = None
    args.time = 1
    args.script_kwargs = None
    args.job_name = None
    args.cc_account = "rrg-dwishart"
    args.cc_email = "a@b.com"
    args.python_version = "3.11"
    args.rdkit_version = "2023.09.5"
    args.no_submit = True
    args.config_list = ["configs/one.yaml", "configs/two.yaml"]

    mod.main(args)

    # Expect two scripts created
    s1 = Path(args.script_dp) / "one.sh"
    s2 = Path(args.script_dp) / "two.sh"
    assert s1.exists()
    assert s2.exists()
