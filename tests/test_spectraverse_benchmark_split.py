"""Tests for preproc_scripts/spectraverse/build_benchmark_split.py."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "preproc_scripts"
    / "spectraverse"
    / "build_benchmark_split.py"
)
module_spec = importlib.util.spec_from_file_location("build_benchmark_split", SCRIPT)
bbs = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(bbs)

N_FOLDS = 10
PER_FOLD = 3  # spectra per fold; each fold has one molecule with 2 spectra and one with 1


def _fold_of_record(i: int) -> int:
    """FOLD of 1-based record ``i`` (records are grouped by fold, 3 per fold)."""
    return (i - 1) // PER_FOLD + 1


@pytest.fixture()
def fixture_files(tmp_path: Path) -> dict[str, Path]:
    """Tiny proc dir + MGF; the spec_df row order is shuffled relative to the MGF."""
    n = N_FOLDS * PER_FOLD
    mol_rows, spec_rows, mgf = [], [], []
    for i in range(1, n + 1):
        fold = _fold_of_record(i)
        mol_id = 2 * (fold - 1) + (0 if (i - 1) % PER_FOLD < 2 else 1)
        spec_rows.append(
            {
                "spec_id": str(i - 1),
                "mol_id": mol_id,
                "dset_spec_id": str(i),
                "group_id": 100 + mol_id,
            }
        )
        mgf.append(
            f"BEGIN IONS\nTITLE=SPECTRAVERSE{i:010d}\nSOURCE=X\nFOLD={fold}\nNUM_PEAKS=0\n"
            "END IONS\n\n"
        )
    for mol_id in range(2 * N_FOLDS):
        mol_rows.append({"mol_id": mol_id, "inchikey_s": f"KEY{mol_id:03d}"})
    spec_df = pd.DataFrame(spec_rows).sample(frac=1.0, random_state=0).reset_index(drop=True)
    proc = tmp_path / "proc"
    proc.mkdir()
    spec_df.to_pickle(proc / "spec_df.pkl")
    pd.DataFrame(mol_rows).to_pickle(proc / "mol_df.pkl")
    mgf_fp = tmp_path / "toy.mgf"
    mgf_fp.write_text("".join(mgf))
    return {"spec": proc / "spec_df.pkl", "mgf": mgf_fp, "tmp": tmp_path}


def _run(files: dict[str, Path], dst: Path, *extra: str) -> int:
    return bbs.main(
        [
            "--spec-fp",
            str(files["spec"]),
            "--mgf-fp",
            str(files["mgf"]),
            "--dst-split-dp",
            str(dst),
            *extra,
        ]
    )


def test_fold_assignment_and_cv0_wrap(fixture_files: dict[str, Path]) -> None:
    dst = fixture_files["tmp"] / "out"
    assert _run(fixture_files, dst) == 0
    # spec_id i-1 belongs to record i; fold = spec_id // 3 + 1
    for cv in range(N_FOLDS):
        val = pd.read_csv(dst / f"cv{cv}" / "val_ids.csv")
        test = pd.read_csv(dst / f"cv{cv}" / "test_ids.csv")
        train = pd.read_csv(dst / f"cv{cv}" / "train_ids.csv")
        val_f = 10 if cv == 0 else cv
        assert set(val["spec_id"] // PER_FOLD + 1) == {val_f}
        assert set(test["spec_id"] // PER_FOLD + 1) == {cv + 1}
        assert set(train["spec_id"] // PER_FOLD + 1) == set(range(1, 11)) - {val_f, cv + 1}
        assert (len(train), len(val), len(test)) == (24, 3, 3)
    cv0_val = pd.read_csv(dst / "cv0" / "val_ids.csv")
    assert list(cv0_val["spec_id"]) == [27, 28, 29]
    cv9_test = pd.read_csv(dst / "cv9" / "test_ids.csv")
    assert list(cv9_test["spec_id"]) == [27, 28, 29]


def test_columns_sorting_and_stats(fixture_files: dict[str, Path]) -> None:
    dst = fixture_files["tmp"] / "out"
    _run(fixture_files, dst, "--folds", "2")
    assert sorted(p.name for p in dst.iterdir()) == ["cv2"]
    spec_df = pd.read_pickle(fixture_files["spec"])
    lookup = spec_df.assign(spec_id=spec_df["spec_id"].astype(int)).set_index("spec_id")
    for name in bbs.SPLITS:
        df = pd.read_csv(dst / "cv2" / f"{name}_ids.csv")
        assert list(df.columns) == ["mol_id", "spec_id", "group_id"]
        assert df["spec_id"].is_monotonic_increasing
        assert (df["mol_id"].to_numpy() == lookup.loc[df["spec_id"], "mol_id"].to_numpy()).all()
        assert (df["group_id"].to_numpy() == lookup.loc[df["spec_id"], "group_id"].to_numpy()).all()
    stats = json.loads((dst / "cv2" / "stats.json").read_text())
    assert stats["num_spectra"] == {"train": 24, "val": 3, "test": 3}
    assert stats["num_molecules"] == {"train": 16, "val": 2, "test": 2}
    assert stats["num_mol_ids"] == {"train": 16, "val": 2, "test": 2}
    assert stats["mgf_md5"] == hashlib.md5(fixture_files["mgf"].read_bytes()).hexdigest()


def test_refuses_nonempty_dst_unless_overwrite(fixture_files: dict[str, Path]) -> None:
    dst = fixture_files["tmp"] / "out"
    dst.mkdir()
    (dst / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        _run(fixture_files, dst)
    assert [p.name for p in dst.iterdir()] == ["keep.txt"]
    # --overwrite still refuses foreign content ...
    with pytest.raises(FileExistsError, match="would not rewrite"):
        _run(fixture_files, dst, "--overwrite", "--folds", "0")
    (dst / "keep.txt").unlink()
    # ... but rewrites a previous build of the same cvs
    assert _run(fixture_files, dst, "--folds", "0") == 0
    assert _run(fixture_files, dst, "--overwrite", "--folds", "0") == 0
    assert (dst / "cv0" / "test_ids.csv").exists()
    # a partial rerun must not leave other cvs of an older build beside the new one
    assert _run(fixture_files, dst, "--overwrite", "--folds", "0", "1") == 0
    with pytest.raises(FileExistsError, match="would not rewrite"):
        _run(fixture_files, dst, "--overwrite", "--folds", "0")


def test_dst_must_be_directory_and_not_the_reference(fixture_files: dict[str, Path]) -> None:
    tmp = fixture_files["tmp"]
    (tmp / "afile").write_text("x")
    with pytest.raises(NotADirectoryError):
        _run(fixture_files, tmp / "afile")
    ref = tmp / "ref"
    assert _run(fixture_files, ref) == 0
    with pytest.raises(ValueError, match="overlaps"):
        _run(fixture_files, ref, "--overwrite", "--check-against", str(ref))
    with pytest.raises(ValueError, match="overlaps"):
        _run(fixture_files, ref / "sub", "--check-against", str(ref))


def test_expect_md5(fixture_files: dict[str, Path]) -> None:
    md5 = hashlib.md5(fixture_files["mgf"].read_bytes()).hexdigest()
    assert _run(fixture_files, fixture_files["tmp"] / "ok", "--expect-md5", md5) == 0
    with pytest.raises(ValueError, match="differs from --expect-md5"):
        _run(fixture_files, fixture_files["tmp"] / "bad", "--expect-md5", "0" * 32)
    assert not (fixture_files["tmp"] / "bad").exists()


def test_dst_split_dp_is_required(fixture_files: dict[str, Path]) -> None:
    with pytest.raises(SystemExit):
        bbs.main(["--spec-fp", str(fixture_files["spec"]), "--mgf-fp", str(fixture_files["mgf"])])


def test_exactly_one_fold_source(fixture_files: dict[str, Path]) -> None:
    with pytest.raises(SystemExit):
        bbs.main(["--spec-fp", str(fixture_files["spec"]), "--dst-split-dp", "x"])


def test_disjointness_assertion_fires(fixture_files: dict[str, Path]) -> None:
    text = fixture_files["mgf"].read_text().replace("FOLD=2\n", "FOLD=3\n", 1)
    fixture_files["mgf"].write_text(text)
    with pytest.raises(ValueError, match="span several folds"):
        _run(fixture_files, fixture_files["tmp"] / "out")
    assert not (fixture_files["tmp"] / "out").exists()


def test_record_count_mismatch(fixture_files: dict[str, Path]) -> None:
    text = fixture_files["mgf"].read_text()
    fixture_files["mgf"].write_text(text.rsplit("BEGIN IONS", 1)[0])
    with pytest.raises(ValueError, match="records but spec_df has"):
        _run(fixture_files, fixture_files["tmp"] / "out")


def test_fold_csv_matches_mgf_and_check_against(fixture_files: dict[str, Path]) -> None:
    tmp = fixture_files["tmp"]
    ref = tmp / "ref"
    assert _run(fixture_files, ref) == 0
    fold_fp = tmp / "toy_fold.csv"
    pd.DataFrame(
        {
            # spec_id deliberately NOT dset_spec_id - 1: the join must use dset_spec_id
            "spec_id": range(N_FOLDS * PER_FOLD, 0, -1),
            "dset_spec_id": range(1, N_FOLDS * PER_FOLD + 1),
            "fold": [_fold_of_record(i + 1) for i in range(N_FOLDS * PER_FOLD)],
        }
    ).to_csv(fold_fp, index=False)
    rc = bbs.main(
        [
            "--spec-fp",
            str(fixture_files["spec"]),
            "--fold-fp",
            str(fold_fp),
            "--dst-split-dp",
            str(tmp / "out2"),
            "--check-against",
            str(ref),
        ]
    )
    assert rc == 0
    for cv in range(N_FOLDS):
        for name in bbs.SPLITS:
            a = (ref / f"cv{cv}" / f"{name}_ids.csv").read_text()
            b = (tmp / "out2" / f"cv{cv}" / f"{name}_ids.csv").read_text()
            assert a == b
    assert "mgf_md5" in (tmp / "out2" / "cv0" / "stats.json").read_text()


def test_check_against_reports_set_and_id_differences(fixture_files: dict[str, Path]) -> None:
    tmp = fixture_files["tmp"]
    ref = tmp / "ref"
    _run(fixture_files, ref)
    # stale mol_id on one row -> sets equal, one mismatch reported
    p = ref / "cv1" / "val_ids.csv"
    df = pd.read_csv(p)
    df.loc[0, "mol_id"] += 99
    df.to_csv(p, index=False)
    spec_df = pd.read_pickle(fixture_files["spec"])
    fold = bbs.assign_folds_from_mgf(spec_df, bbs.read_mgf_folds(fixture_files["mgf"])[0])
    built = {1: bbs.build_cv(spec_df, fold, 1)}
    ok, lines = bbs.check_against(built, ref)
    assert ok
    assert any("cv1/val" in ln and "differing mol_id 1," in ln for ln in lines)
    # swapped ids -> sets differ
    other = {1: bbs.build_cv(spec_df, fold, 2)}
    ok, lines = bbs.check_against(other, ref)
    assert not ok
    assert any("DIFFER" in ln for ln in lines)


def test_repeated_title_rejected(fixture_files: dict[str, Path]) -> None:
    text = (
        fixture_files["mgf"].read_text().replace("SPECTRAVERSE0000000002", "SPECTRAVERSE0000000009")
    )
    fixture_files["mgf"].write_text(text)
    with pytest.raises(ValueError, match="repeats TITLE number 9"):
        bbs.read_mgf_folds(fixture_files["mgf"])


def test_sparse_titles_join_on_title_number(fixture_files: dict[str, Path]) -> None:
    """v1.0.3 TITLE numbers have gaps; folds still follow the TITLE, not the record position."""
    tmp = fixture_files["tmp"]
    ref = tmp / "ref"
    assert _run(fixture_files, ref) == 0
    # renumber every TITLE n -> 2n + 5 in both the MGF and spec_df
    n = N_FOLDS * PER_FOLD
    text = fixture_files["mgf"].read_text()
    for i in range(n, 0, -1):
        text = text.replace(f"SPECTRAVERSE{i:010d}", f"SPECTRAVERSE{2 * i + 5:010d}")
    fixture_files["mgf"].write_text(text)
    spec_df = pd.read_pickle(fixture_files["spec"])
    spec_df["dset_spec_id"] = (2 * spec_df["dset_spec_id"].astype(int) + 5).astype(str)
    spec_df.to_pickle(fixture_files["spec"])
    assert _run(fixture_files, tmp / "out", "--check-against", str(ref)) == 0
    for cv in range(N_FOLDS):
        for name in bbs.SPLITS:
            a = (ref / f"cv{cv}" / f"{name}_ids.csv").read_text()
            assert a == (tmp / "out" / f"cv{cv}" / f"{name}_ids.csv").read_text()
    # a TITLE outside spec_df's dset_spec_id set is refused
    spec_df.loc[0, "dset_spec_id"] = "1"
    with pytest.raises(ValueError, match="TITLE numbers differ"):
        bbs.assign_folds_from_mgf(spec_df, bbs.read_mgf_folds(fixture_files["mgf"])[0])


def test_source_filter_reads_sources_by_title(tmp_path: Path) -> None:
    """``read_mgf_sources`` keys SOURCE by TITLE number, so sparse titles map correctly."""
    sfs_path = Path(__file__).resolve().parents[1] / "preproc_scripts/spectraverse/source_filter_split.py"
    spec = importlib.util.spec_from_file_location("source_filter_split", sfs_path)
    sfs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sfs)
    mgf = tmp_path / "s.mgf"
    mgf.write_text(
        "BEGIN IONS\nTITLE=SPECTRAVERSE0000000001\nSOURCE=GNPS\nEND IONS\n\n"
        "BEGIN IONS\nTITLE=SPECTRAVERSE0000000003\nSOURCE=MS-DIAL\nEND IONS\n"
    )
    assert sfs.read_mgf_sources(str(mgf)) == {1: "GNPS", 3: "MS-DIAL"}
    mgf.write_text(mgf.read_text().replace("0000000003", "0000000001"))
    with pytest.raises(ValueError, match="repeats TITLE number 1"):
        sfs.read_mgf_sources(str(mgf))


def test_fold_csv_wrong_columns_or_ids_rejected(fixture_files: dict[str, Path]) -> None:
    spec_df = pd.read_pickle(fixture_files["spec"])
    n = N_FOLDS * PER_FOLD
    fold_fp = fixture_files["tmp"] / "f.csv"
    pd.DataFrame({"spec_id": range(n), "fold": [1] * n}).to_csv(fold_fp, index=False)
    with pytest.raises(ValueError, match="must have columns"):
        bbs.assign_folds_from_csv(spec_df, fold_fp)
    pd.DataFrame({"spec_id": range(n), "dset_spec_id": range(2, n + 2), "fold": [1] * n}).to_csv(
        fold_fp, index=False
    )
    with pytest.raises(ValueError, match="dset_spec_id set differs"):
        bbs.assign_folds_from_csv(spec_df, fold_fp)


def test_mgf_error_messages(fixture_files: dict[str, Path]) -> None:
    mgf = fixture_files["mgf"]
    text = mgf.read_text()
    mgf.write_text(text.replace("FOLD=1\n", "FOLD=1.0\n", 1))
    with pytest.raises(ValueError, match="record 1: FOLD is not an integer"):
        bbs.read_mgf_folds(mgf)
    mgf.write_text(text.replace("FOLD=1\n", "FOLD_MCES=1\n", 1))
    with pytest.raises(ValueError, match="legacy INDEX/FOLD_"):
        bbs.read_mgf_folds(mgf)
