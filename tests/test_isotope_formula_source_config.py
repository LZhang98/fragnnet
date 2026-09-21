"""Config wiring for the isotope-cleanup fragment-composition source.

The detector itself is covered in ``test_isotope_detection.py``; this file covers which source
a dataset builds from ``spec_params``, and that the superseded keys fail loudly instead of
being ignored.
"""

import numpy as np
import pandas as pd
import pytest

from fragnnet.dataset.base_dataset import ISOTOPE_FORMULA_SOURCES, BaseDataset
from fragnnet.dataset.spec_mol_frag_dataset import wire_isotope_dag_paths


def _dataset(**overrides) -> BaseDataset:
    dataset = BaseDataset.__new__(BaseDataset)
    dataset.spec_params = {
        "remove_isotope_peaks": True,
        "remove_isotope_use_group_f": False,
        "remove_isotope_formula_source": "none",
        "remove_isotope_candidate_rule": "median",
        "remove_isotope_subformula_mz_tol": None,
        "remove_isotope_use_dag_formula": False,
        "remove_isotope_dag_dp": None,
        "remove_isotope_dag_compressed": False,
        "remove_isotope_dag_mz_tol": 0.015,
        "remove_isotope_protect_known_mono": False,
        "remove_isotope_protect_dag_mono": False,
        **overrides,
    }
    dataset.um_spec_df = pd.DataFrame()
    dataset.mol_df = pd.DataFrame({"mol_id": [1], "formula": ["C9H8O4"]}).set_index(
        "mol_id", drop=False
    )
    dataset._isotope_dag_formula_cache = {}
    return dataset


def test_default_source_is_none():
    """The feature ships off, and off means the m/z-scaled proxy, not a DAG read."""
    dataset = _dataset()

    assert dataset._get_isotope_formula_source(1, "C9H8O4", "[M+H]+", "FT") is None


def test_subformula_source_needs_no_dag():
    """The point of the source: a composition for every model, with no per-molecule artifact."""
    dataset = _dataset(remove_isotope_formula_source="subformula")

    source = dataset._get_isotope_formula_source(1, "C9H8O4", "[M+H]+", "FT")

    assert source is not None
    assert {"C": 7, "H": 5, "O": 2} in source(121.02841)


def test_subformula_window_defaults_to_instrument_tolerance():
    """Blank tolerance means FT's 4 mDa here; QTOF's wider window admits more candidates."""
    dataset = _dataset(remove_isotope_formula_source="subformula")

    ft_source = dataset._get_isotope_formula_source(1, "C9H8O4", "[M+H]+", "FT")
    explicit_source = _dataset(
        remove_isotope_formula_source="subformula",
        remove_isotope_subformula_mz_tol=0.05,
    )._get_isotope_formula_source(1, "C9H8O4", "[M+H]+", "FT")

    assert len(explicit_source(121.02841)) > len(ft_source(121.02841))


def test_subformula_source_falls_back_when_the_adduct_is_unsupported():
    """An adduct the budgets do not cover costs the composition, not the spectrum."""
    dataset = _dataset(remove_isotope_formula_source="subformula")

    assert dataset._get_isotope_formula_source(1, "C9H8O4", "[M+Xx]+", "FT") is None


def test_subformula_source_needs_a_formula():
    dataset = _dataset(remove_isotope_formula_source="subformula")

    assert dataset._get_isotope_formula_source(1, None, "[M+H]+", "FT") is None


def test_dag_source_without_a_dag_directory_is_none():
    dataset = _dataset(remove_isotope_formula_source="dag")

    assert dataset._get_isotope_formula_source(1, "C9H8O4", "[M+H]+", "FT") is None


def test_dag_source_reads_the_cached_dag_arrays():
    dataset = _dataset(remove_isotope_formula_source="dag", remove_isotope_dag_dp="unused")
    dataset._isotope_dag_formula_cache[1] = (
        np.array([121.02841]),
        np.array(["C7H5O2"], dtype=object),
    )

    source = dataset._get_isotope_formula_source(1, "C9H8O4", "[M+H]+", "FT")

    assert source(121.02841) == [{"C": 7, "H": 5, "O": 2}]


def test_unknown_source_is_rejected_at_setup():
    dataset = _dataset(remove_isotope_formula_source="oracle")

    with pytest.raises(ValueError, match="remove_isotope_formula_source must be one of"):
        dataset._setup_isotope_peak_cleanup()


@pytest.mark.parametrize(
    ("deprecated", "match"),
    [
        ("remove_isotope_use_dag_formula", "remove_isotope_formula_source: dag"),
        ("remove_isotope_protect_dag_mono", "remove_isotope_protect_known_mono"),
    ],
)
def test_superseded_keys_fail_loudly(deprecated, match):
    """They stay declared so old configs resolve; switching one on must not be ignored."""
    dataset = _dataset(**{deprecated: True})

    with pytest.raises(ValueError, match=match):
        dataset._setup_isotope_peak_cleanup()


def test_disabled_cleanup_skips_validation():
    """Nothing is validated for a feature that does not run; that is why old configs resolve."""
    dataset = _dataset(remove_isotope_peaks=False, remove_isotope_formula_source="oracle")

    dataset._setup_isotope_peak_cleanup()

    assert dataset.group_coiso_fraction_by_k == {}


def test_source_names_are_the_declared_set():
    assert ISOTOPE_FORMULA_SOURCES == ("none", "subformula", "dag")


def test_subformula_source_removes_an_isotope_peak_end_to_end():
    """The whole dataset path: spec_params -> source -> detector -> peaks dropped, no DAG."""
    import torch as th

    from fragnnet.utils.formula_utils import C13_SPACING, parse_formula
    from fragnnet.utils.isotope_utils import _expected_mk_ratio

    formula, frag_formula = "C10H22", "C5H11"
    frag_mz = 71.086
    r_prec = _expected_mk_ratio(parse_formula(formula), 1.0, k=1)
    r_frag = _expected_mk_ratio(parse_formula(frag_formula), 1.0, k=1)
    observed_ratio = r_frag / (1.0 + (r_prec - r_frag))

    dataset = _dataset(
        remove_isotope_formula_source="subformula",
        remove_isotope_max_isotope=2,
        remove_regular_isotope_peaks=False,
        remove_coisolation_peaks=True,
        remove_isotope_regular_mz_tol=None,
        remove_isotope_min_coiso_fraction=0.05,
        remove_isotope_max_leak_factor=1.5,
        remove_isotope_precursor_envelope_lo=0.2,
        remove_isotope_precursor_envelope_hi=5.0,
    )
    dataset.mol_df = pd.DataFrame({"mol_id": [1], "formula": [formula]}).set_index(
        "mol_id", drop=False
    )
    dataset.group_coiso_fraction_by_k = {}
    spec_entry = pd.Series(
        {
            "group_id": 1,
            "mol_id": 1,
            "prec_type": "[M+H]+",
            "inst_type": "FT",
            "prec_mz": 143.179,
        }
    )
    mzs = th.tensor([frag_mz, frag_mz + C13_SPACING], dtype=th.float64)
    ints = th.tensor([2000.0, observed_ratio * 2000.0], dtype=th.float64)

    kept_mzs, kept_ints = dataset._remove_isotope_peaks_from_spec(spec_entry, mzs, ints)

    assert kept_mzs.tolist() == [frag_mz]
    assert kept_ints.tolist() == [2000.0]


def test_unknown_candidate_rule_is_rejected_at_setup():
    """Otherwise it surfaces only on the first spectrum whose peak draws two candidates."""
    dataset = _dataset(remove_isotope_candidate_rule="mean")

    with pytest.raises(ValueError, match="remove_isotope_candidate_rule must be one of"):
        dataset._setup_isotope_peak_cleanup()


def test_dag_source_without_a_path_is_rejected_at_setup():
    """A blank path would otherwise mean an empty source and a silent m/z-proxy answer."""
    dataset = _dataset(remove_isotope_formula_source="dag", remove_isotope_dag_dp=None)

    with pytest.raises(ValueError, match="needs remove_isotope_dag_dp"):
        dataset._setup_isotope_peak_cleanup()


class TestWireIsotopeDagPaths:
    """FraGNNet is the only dataset that can fill the DAG path in, so it has to actually do it."""

    SPEC_PARAMS = {
        "remove_isotope_peaks": True,
        "remove_isotope_formula_source": "dag",
        "remove_isotope_dag_dp": None,
        "remove_isotope_dag_compressed": False,
    }
    FRAG_PARAMS = {"compressed": True}

    def test_dag_source_gets_this_datasets_dags(self):
        wired = wire_isotope_dag_paths(self.SPEC_PARAMS, "data/frag/x/dags", self.FRAG_PARAMS)

        assert wired["remove_isotope_dag_dp"] == "data/frag/x/dags"
        # How the DAGs are stored is a property of the DAGs the path now points at.
        assert wired["remove_isotope_dag_compressed"] is True

    def test_an_explicit_path_is_left_alone(self):
        spec_params = {**self.SPEC_PARAMS, "remove_isotope_dag_dp": "data/frag/other/dags"}

        wired = wire_isotope_dag_paths(spec_params, "data/frag/x/dags", self.FRAG_PARAMS)

        assert wired["remove_isotope_dag_dp"] == "data/frag/other/dags"
        assert wired["remove_isotope_dag_compressed"] is False

    @pytest.mark.parametrize(
        "overrides",
        [
            {"remove_isotope_peaks": False},
            {"remove_isotope_formula_source": "subformula"},
            {"remove_isotope_formula_source": "none"},
        ],
    )
    def test_nothing_is_wired_when_the_dag_source_is_not_asked_for(self, overrides):
        wired = wire_isotope_dag_paths(
            {**self.SPEC_PARAMS, **overrides}, "data/frag/x/dags", self.FRAG_PARAMS
        )

        assert wired["remove_isotope_dag_dp"] is None

    def test_the_input_is_not_mutated(self):
        spec_params = dict(self.SPEC_PARAMS)

        wire_isotope_dag_paths(spec_params, "data/frag/x/dags", self.FRAG_PARAMS)

        assert spec_params["remove_isotope_dag_dp"] is None


@pytest.mark.parametrize("mz_tol", [0.0, -0.004])
def test_bad_subformula_tolerance_is_rejected_at_setup(mz_tol):
    """A bad tolerance is a config error; the per-spectrum fallback must not absorb it."""
    dataset = _dataset(
        remove_isotope_formula_source="subformula",
        remove_isotope_subformula_mz_tol=mz_tol,
    )

    with pytest.raises(ValueError, match="remove_isotope_subformula_mz_tol must be positive"):
        dataset._setup_isotope_peak_cleanup()


def test_protect_known_mono_keeps_a_peak_the_subformula_source_explains():
    """The highest-precision arm, reachable from config: an H-gain alternative wins the peak.

    72.0894 is one 13C step above the C5H11 fragment, and also within 4 mDa of C5H12 - the
    H-gain ion the FT window is meant to separate. Protection is what that ambiguity buys.
    """
    import torch as th

    from fragnnet.utils.formula_utils import C13_SPACING, parse_formula
    from fragnnet.utils.isotope_utils import _expected_mk_ratio

    formula, frag_mz = "C10H22", 71.086
    r_prec = _expected_mk_ratio(parse_formula(formula), 1.0, k=1)
    r_frag = _expected_mk_ratio(parse_formula("C5H11"), 1.0, k=1)
    observed_ratio = r_frag / (1.0 + (r_prec - r_frag))

    spec_entry = pd.Series(
        {
            "group_id": 1,
            "mol_id": 1,
            "prec_type": "[M+H]+",
            "inst_type": "FT",
            "prec_mz": 143.179,
        }
    )
    mzs = th.tensor([frag_mz, frag_mz + C13_SPACING], dtype=th.float64)
    ints = th.tensor([2000.0, observed_ratio * 2000.0], dtype=th.float64)

    def run(protect: bool):
        dataset = _dataset(
            remove_isotope_formula_source="subformula",
            remove_isotope_max_isotope=2,
            remove_regular_isotope_peaks=False,
            remove_coisolation_peaks=True,
            remove_isotope_regular_mz_tol=None,
            remove_isotope_min_coiso_fraction=0.05,
            remove_isotope_max_leak_factor=1.5,
            remove_isotope_precursor_envelope_lo=0.2,
            remove_isotope_precursor_envelope_hi=5.0,
            remove_isotope_protect_known_mono=protect,
        )
        dataset.mol_df = pd.DataFrame({"mol_id": [1], "formula": [formula]}).set_index(
            "mol_id", drop=False
        )
        dataset.group_coiso_fraction_by_k = {}
        return dataset._remove_isotope_peaks_from_spec(spec_entry, mzs, ints)[0]

    assert run(protect=False).tolist() == [frag_mz]
    assert len(run(protect=True)) == 2


def test_the_dataset_hands_the_detector_its_configured_rule_and_protection(monkeypatch):
    """The knobs are useless if they stop at `spec_params`."""
    import torch as th

    from fragnnet.dataset import base_dataset as base_dataset_module

    captured = {}

    def fake_cleanup(*_args, **kwargs):
        captured.update(kwargs)
        return np.zeros(2, dtype=bool)

    monkeypatch.setattr(
        base_dataset_module, "detect_isotope_peaks_for_training_cleanup", fake_cleanup
    )
    dataset = _dataset(
        remove_isotope_formula_source="subformula",
        remove_isotope_candidate_rule="all",
        remove_isotope_protect_known_mono=True,
        remove_isotope_max_isotope=2,
        remove_regular_isotope_peaks=False,
        remove_coisolation_peaks=True,
        remove_isotope_regular_mz_tol=None,
        remove_isotope_min_coiso_fraction=0.05,
        remove_isotope_max_leak_factor=1.5,
        remove_isotope_precursor_envelope_lo=0.2,
        remove_isotope_precursor_envelope_hi=5.0,
    )
    dataset.mol_df = pd.DataFrame({"mol_id": [1], "formula": ["C10H22"]}).set_index(
        "mol_id", drop=False
    )
    dataset.group_coiso_fraction_by_k = {}
    spec_entry = pd.Series(
        {
            "group_id": 1,
            "mol_id": 1,
            "prec_type": "[M+H]+",
            "inst_type": "FT",
            "prec_mz": 143.179,
        }
    )

    dataset._remove_isotope_peaks_from_spec(
        spec_entry,
        th.tensor([71.086, 72.0894], dtype=th.float64),
        th.tensor([2000.0, 50.0], dtype=th.float64),
    )

    assert captured["candidate_rule"] == "all"
    assert captured["protect_known_mono"] is True
    assert captured["formula_source"](71.086) == [{"C": 5, "H": 11}]


def test_wiring_the_dag_path_requires_knowing_how_the_dags_are_stored():
    """`frag_params['compressed']` is required, not guessed: reading them wrong is silent."""
    spec_params = {
        "remove_isotope_peaks": True,
        "remove_isotope_formula_source": "dag",
        "remove_isotope_dag_dp": None,
        "remove_isotope_dag_compressed": False,
    }

    with pytest.raises(KeyError, match="compressed"):
        wire_isotope_dag_paths(spec_params, "data/frag/x/dags", {})
