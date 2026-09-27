import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    fp = repo_root / "scripts" / "setup_job_cc.py"
    spec = importlib.util.spec_from_file_location("setup_job_cc", str(fp))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return load_module()


def make_args(**kwargs) -> argparse.Namespace:
    """Return a Namespace with sensible defaults, overridden by kwargs."""
    defaults = {
        "gpu_profile": None,
        "gpu": 1,
        "cpu": 8,
        "mem": 128,
        "gres_gpu": None,
        "gpus_per_node": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# _normalize_profile_name
# ---------------------------------------------------------------------------


class TestNormalizeProfileName:
    def test_lowercases(self, mod):
        assert mod._normalize_profile_name("H100-80GB") == "h100-80gb"

    def test_strips_leading_trailing_whitespace(self, mod):
        assert mod._normalize_profile_name("  a100-40gb  ") == "a100-40gb"

    def test_removes_internal_spaces(self, mod):
        # e.g. user types "fir: h100-80gb"
        assert mod._normalize_profile_name("fir: h100-80gb") == "fir:h100-80gb"

    def test_already_normalized_unchanged(self, mod):
        assert mod._normalize_profile_name("narval:a100-40gb") == "narval:a100-40gb"

    def test_empty_string(self, mod):
        assert mod._normalize_profile_name("") == ""


# ---------------------------------------------------------------------------
# GPU_PROFILES structure
# ---------------------------------------------------------------------------

REQUIRED_PROFILE_KEYS = {"cluster", "gpu_label", "gres_gpu", "cpu", "mem", "price", "notes"}


class TestGpuProfilesStructure:
    def test_all_profiles_have_required_keys(self, mod):
        for name, profile in mod.GPU_PROFILES.items():
            missing = REQUIRED_PROFILE_KEYS - profile.keys()
            assert not missing, f"Profile '{name}' missing keys: {missing}"

    def test_cpu_and_mem_are_positive(self, mod):
        for name, profile in mod.GPU_PROFILES.items():
            assert profile["cpu"] > 0, f"Profile '{name}' has non-positive cpu"
            assert profile["mem"] > 0, f"Profile '{name}' has non-positive mem"

    def test_gres_starts_with_gpu(self, mod):
        for name, profile in mod.GPU_PROFILES.items():
            assert profile["gres_gpu"].startswith("gpu"), (
                f"Profile '{name}' gres_gpu does not start with 'gpu': {profile['gres_gpu']}"
            )

    def test_known_fir_h100_80gb_values(self, mod):
        p = mod.GPU_PROFILES["fir:h100-80gb"]
        assert p["cluster"] == "fir"
        assert p["cpu"] == 12
        assert p["mem"] == 280
        assert p["gres_gpu"] == "gpu:h100:1"

    def test_known_narval_a100_40gb_values(self, mod):
        p = mod.GPU_PROFILES["narval:a100-40gb"]
        assert p["cluster"] == "narval"
        assert p["cpu"] == 12
        assert p["mem"] == 124
        assert p["gres_gpu"] == "gpu:a100:1"


# ---------------------------------------------------------------------------
# GPU_PROFILE_ALIASES
# ---------------------------------------------------------------------------


class TestGpuProfileAliases:
    def test_all_aliases_point_to_valid_profiles(self, mod):
        for alias, target in mod.GPU_PROFILE_ALIASES.items():
            assert target in mod.GPU_PROFILES, (
                f"Alias '{alias}' points to unknown profile '{target}'"
            )

    def test_h100_80gb_alias(self, mod):
        assert mod.GPU_PROFILE_ALIASES["h100-80gb"] == "fir:h100-80gb"

    def test_a100_40gb_alias(self, mod):
        assert mod.GPU_PROFILE_ALIASES["a100-40gb"] == "narval:a100-40gb"

    def test_a6000_alias(self, mod):
        assert mod.GPU_PROFILE_ALIASES["a6000"] == "vulcan:l40s"


# ---------------------------------------------------------------------------
# apply_gpu_profile
# ---------------------------------------------------------------------------


class TestApplyGpuProfile:
    def test_no_profile_is_noop(self, mod):
        args = make_args(gpu_profile=None, cpu=4, mem=64)
        mod.apply_gpu_profile(args)
        assert args.cpu == 4
        assert args.mem == 64

    def test_full_profile_key_sets_resources(self, mod):
        args = make_args(gpu_profile="fir:h100-80gb", gpu=0)
        mod.apply_gpu_profile(args)
        expected = mod.GPU_PROFILES["fir:h100-80gb"]
        assert args.cpu == expected["cpu"]
        assert args.mem == expected["mem"]
        assert args.gres_gpu == expected["gres_gpu"]

    def test_alias_resolves_to_correct_profile(self, mod):
        args = make_args(gpu_profile="a100-40gb", gpu=0)
        mod.apply_gpu_profile(args)
        expected = mod.GPU_PROFILES["narval:a100-40gb"]
        assert args.cpu == expected["cpu"]
        assert args.mem == expected["mem"]
        assert args.gres_gpu == expected["gres_gpu"]

    def test_case_insensitive_input(self, mod):
        args = make_args(gpu_profile="FIR:H100-80GB", gpu=0)
        mod.apply_gpu_profile(args)
        expected = mod.GPU_PROFILES["fir:h100-80gb"]
        assert args.cpu == expected["cpu"]
        assert args.gres_gpu == expected["gres_gpu"]

    def test_whitespace_in_input(self, mod):
        args = make_args(gpu_profile="  narval:a100-40gb  ", gpu=0)
        mod.apply_gpu_profile(args)
        expected = mod.GPU_PROFILES["narval:a100-40gb"]
        assert args.cpu == expected["cpu"]

    def test_unknown_profile_raises_value_error(self, mod):
        args = make_args(gpu_profile="nonexistent:gpu")
        with pytest.raises(ValueError, match="Unknown gpu profile"):
            mod.apply_gpu_profile(args)

    def test_gpu_set_to_1_when_zero(self, mod):
        """gpu=0 should be bumped to 1 after applying a profile."""
        args = make_args(gpu_profile="fir:h100-1g.10gb", gpu=0)
        mod.apply_gpu_profile(args)
        assert args.gpu == 1

    def test_gpu_not_overwritten_when_already_positive(self, mod):
        """Pre-existing positive gpu value must be preserved."""
        args = make_args(gpu_profile="fir:h100-1g.10gb", gpu=2)
        mod.apply_gpu_profile(args)
        assert args.gpu == 2

    def test_mig_slice_sets_correct_gres(self, mod):
        args = make_args(gpu_profile="narval:a100-1g.5gb", gpu=0)
        mod.apply_gpu_profile(args)
        assert args.gres_gpu == "gpu:a100_1g.5gb:1"

    def test_all_profiles_apply_without_error(self, mod):
        """Smoke test: every declared profile applies cleanly."""
        for name in mod.GPU_PROFILES:
            args = make_args(gpu_profile=name, gpu=0)
            mod.apply_gpu_profile(args)

    def test_alias_case_insensitive(self, mod):
        """Aliases should also be resolved case-insensitively."""
        args = make_args(gpu_profile="A100-40GB", gpu=0)
        mod.apply_gpu_profile(args)
        expected = mod.GPU_PROFILES["narval:a100-40gb"]
        assert args.cpu == expected["cpu"]


def test_channel_selection_audit_stages_checkpoint_and_preserves_output(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    config_dp = project / "config"
    data_dp = project / "data"
    checkpoint_dp = project / "saved_ckpts"
    config_dp.mkdir(parents=True)
    data_dp.mkdir()
    checkpoint_dp.mkdir()
    (data_dp / "spec.pkl").touch()
    (data_dp / "mol.pkl").touch()
    (data_dp / "split").mkdir()
    (checkpoint_dp / "model.ckpt").touch()
    (config_dp / "template.yml").write_text(
        "model_type: masster\n"
        "finetune: False\n"
        "spec_fp: data/spec.pkl\n"
        "mol_fp: data/mol.pkl\n"
        "split_dp: data/split\n"
    )
    (config_dp / "custom.yml").write_text("model_type: masster\n")

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "setup_job_cc.py"),
            "--job_name=channel_audit",
            "--python_script_fp=./scripts/eval/masster_ion_h_top_k_sweep.py",
            "--frag_gnn_dp=./",
            "--script_dp=slurm_scripts",
            "--job_dp=jobs",
            "--script_kwargs",
            "model_ckpt=saved_ckpts/model.ckpt",
            "custom_fp=config/custom.yml",
            "fragment_relative_thresholds=0 0.01",
            "ion_h_top_ks=0 4",
            "joint_relative_thresholds=",
            "output_fp=output/eval/result.csv",
            "device=cuda",
            "wandb_mode=disabled",
        ],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )

    script = (project / "slurm_scripts" / "channel_audit.sh").read_text()
    assert (
        "rsync -avL --relative ${FRAGNNET_DP}/./saved_ckpts/model.ckpt ${WORKING_DIR}"
        in script
    )
    assert "if [[ ! -f ${WORKING_DIR}/saved_ckpts/model.ckpt ]]; then" in script
    assert "ERROR: checkpoint was not staged" in script
    assert "--model_ckpt saved_ckpts/model.ckpt" in script
    assert "--fragment_relative_thresholds 0 0.01" in script
    assert "--ion_h_top_ks 0 4" in script
    assert f"mkdir -p {project}/output/eval" in script
    assert (
        "if [[ -f ${WORKING_DIR}/output/eval/result.csv ]]; then\n"
        f"    rsync -av ${{WORKING_DIR}}/output/eval/result.csv "
        f"{project}/output/eval/result.csv" in script
    )
    assert "CHANNEL_AUDIT_STATUS=$?" in script
    assert 'exit "${CHANNEL_AUDIT_STATUS}"' in script


def test_job_script_stages_the_template(tmp_path):
    """`config/` is never staged wholesale, so the template has to be copied by name.

    With one template there is nothing else for validation to read, so what the node sees and
    what the submitting machine saw are the same file.
    """
    repo_root = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    config_dp = project / "config"
    data_dp = project / "data"
    config_dp.mkdir(parents=True)
    data_dp.mkdir()
    (data_dp / "spec.pkl").touch()
    (data_dp / "mol.pkl").touch()
    (data_dp / "split").mkdir()
    (data_dp / "dags").mkdir()
    (config_dp / "template.yml").write_text(
        "model_type: frag_gnn\n"
        "finetune: False\n"
        "spec_fp: data/spec.pkl\n"
        "mol_fp: data/mol.pkl\n"
        "split_dp: data/split\n"
        "frag_dp: data/dags\n"
    )
    (config_dp / "custom.yml").write_text("model_type: frag_gnn\n")

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "setup_job_cc.py"),
            "--job_name=template_staging",
            "--python_script_fp=./scripts/run_pl_model_fit.py",
            "--frag_gnn_dp=./",
            "--script_dp=slurm_scripts",
            "--job_dp=jobs",
            "--script_kwargs",
            "custom_fp=config/custom.yml",
            "wandb_mode=disabled",
        ],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )

    script = (project / "slurm_scripts" / "template_staging.sh").read_text()
    assert "rsync -av --relative ${FRAGNNET_DP}/./config/template.yml ${WORKING_DIR}" in script
    assert "*_template.yml" not in script


def test_job_script_stops_when_the_venv_is_missing(tmp_path):
    """A failed `virtualenv` must end the job before any bare `pip` can write to ~/.local."""
    repo_root = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    config_dp = project / "config"
    data_dp = project / "data"
    config_dp.mkdir(parents=True)
    data_dp.mkdir()
    (data_dp / "spec.pkl").touch()
    (data_dp / "mol.pkl").touch()
    (data_dp / "split").mkdir()
    (data_dp / "dags").mkdir()
    (config_dp / "template.yml").write_text(
        "model_type: frag_gnn\n"
        "finetune: False\n"
        "spec_fp: data/spec.pkl\n"
        "mol_fp: data/mol.pkl\n"
        "split_dp: data/split\n"
        "frag_dp: data/dags\n"
    )
    (config_dp / "custom.yml").write_text("model_type: frag_gnn\n")

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "setup_job_cc.py"),
            "--job_name=venv_guard",
            "--python_script_fp=./scripts/run_pl_model_fit.py",
            "--frag_gnn_dp=./",
            "--script_dp=slurm_scripts",
            "--job_dp=jobs",
            "--script_kwargs",
            "custom_fp=config/custom.yml",
            "wandb_mode=disabled",
        ],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )

    script = (project / "slurm_scripts" / "venv_guard.sh").read_text()
    assert "export PYTHONNOUSERSITE=1" in script
    assert "source FRAG-GNN/bin/activate || exit 1" in script
    first_pip = script.index("pip install")
    assert script.index("export PYTHONNOUSERSITE=1") < first_pip
    assert script.index("source FRAG-GNN/bin/activate || exit 1") < first_pip
