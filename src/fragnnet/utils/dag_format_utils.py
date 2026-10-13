"""Fragment-DAG files, one per molecule: file naming, the pickle writer and the reader.

Core code (``base_dataset``) reads these per-molecule DAG files, so this layout lives in core with
its reader and writer in one place. The packed ``dags.h5`` format, which only FraGNNet reads and
writes, is in ``models/fragnnet/dag/storage.py``.
"""

import bz2
import gzip
import os
import pickle
import tarfile
from zipfile import ZipFile


def get_frag_name(mol_id: str | int, is_compressed: bool) -> str:
    """Return the canonical filename for a fragment DAG pickle file.

    Args:
        mol_id: Molecule identifier (int or numeric string).
        is_compressed: Whether the file is bz2-compressed.

    Returns:
        Filename of the form ``{mol_id}.pkl[.bz2]``.
    """
    name = f"{int(mol_id)}.pkl"
    if is_compressed:
        name += ".bz2"
    return name


def get_frag_fp(mol_id: str | int, frag_dp: str, is_compressed: bool) -> str:
    """Return the full path for a fragment DAG pickle file.

    Args:
        mol_id: Molecule identifier (int or numeric string).
        frag_dp: Directory containing fragment files.
        is_compressed: Whether the file is bz2-compressed.

    Returns:
        Full file path using the canonical naming convention.
    """
    return os.path.join(frag_dp, get_frag_name(mol_id, is_compressed))


def _dump_pickle_file(fp: str, payload: dict) -> None:
    """Write a pickled payload using compression inferred from file suffix."""
    if fp.endswith((".pkl", ".pickle")):
        with open(fp, "wb") as pf:
            pickle.dump(payload, pf, protocol=pickle.HIGHEST_PROTOCOL)
        return

    if fp.endswith((".pkl.gz", ".pickle.gz")):
        with gzip.open(fp, "wb") as pf:
            pickle.dump(payload, pf, protocol=pickle.HIGHEST_PROTOCOL)
        return

    if fp.endswith((".pkl.bz2", ".pickle.bz2")):
        with bz2.open(fp, "wb") as pf:
            pickle.dump(payload, pf, protocol=pickle.HIGHEST_PROTOCOL)
        return

    raise ValueError(
        f"Unsupported pickle output extension for '{fp}'. "
        "Expected .pkl/.pickle with optional .gz or .bz2"
    )


def _load_pickle_file(fp: str):
    """Read a pickled payload using compression inferred from file suffix."""
    if fp.endswith((".pkl", ".pickle")):
        with open(fp, "rb") as pf:
            return pickle.load(pf)

    if fp.endswith((".pkl.gz", ".pickle.gz")):
        with gzip.open(fp, "rb") as pf:
            return pickle.load(pf)

    if fp.endswith((".pkl.bz2", ".pickle.bz2")):
        with bz2.open(fp, "rb") as pf:
            return pickle.load(pf)

    raise ValueError(
        f"Unsupported pickle input extension for '{fp}'. "
        "Expected .pkl/.pickle with optional .gz or .bz2"
    )


def get_dag_output_path(
    dag_dp: str,
    mol_id: str | int,
    compress_dags: bool,
    compress_format: str,
) -> str:
    """Build output path for one DAG artifact.

    The base filename is determined by :func:`get_frag_name` (canonical format
    ``{mol_id}.pkl``).  Compression with ``"gz"`` appends ``.gz``; ``"bz2"``
    appends ``.bz2`` (equivalent to passing ``is_compressed=True`` to
    :func:`get_frag_name`).

    Args:
        dag_dp: DAG output directory.
        mol_id: Molecule identifier.
        compress_dags: Whether per-file DAG outputs should be compressed.
        compress_format: Compression format, either ``"gz"`` or ``"bz2"``.

    Returns:
        Full output path for the DAG file.

    Raises:
        ValueError: If ``compress_format`` is invalid when compression is enabled.
    """
    if not compress_dags:
        return get_frag_fp(mol_id, dag_dp, is_compressed=False)

    if compress_format == "gz":
        return os.path.join(dag_dp, f"{int(mol_id)}.pkl.gz")
    if compress_format == "bz2":
        return get_frag_fp(mol_id, dag_dp, is_compressed=True)

    raise ValueError(
        f"Invalid compression format '{compress_format}'. Expected one of: 'gz', 'bz2'."
    )


def dump_dag_pickle(fp: str, dag_d: dict) -> None:
    """Write a DAG dictionary to disk with extension-aware compression.

    Args:
        fp: Output path ending with ``.pkl``, ``.pkl.gz``, or ``.pkl.bz2``.
        dag_d: DAG payload dictionary.

    Raises:
        ValueError: If file suffix is unsupported.
    """
    _dump_pickle_file(fp, dag_d)


def save_frag_d(frag_d: dict, mol_id: str, frag_dp: str, is_compressed: bool = False) -> None:
    """save frag_d use pickle if is_compressed save as .pbz
    Args:
        frag_d (dict): frag_d to save
        mol_id (str): mol_id to save as
        frag_dp (str): directory to save to
        is_compressed (bool, optional): whether to compress the file. Defaults to False.
    """

    fp = get_frag_fp(mol_id, frag_dp, is_compressed)
    _dump_pickle_file(fp, frag_d)


def _legacy_frag_names(mol_id: str | int, is_compressed: bool) -> list[str]:
    """Return legacy filenames to try for backward compatibility.

    Covers two previous naming conventions:

    - ``{mol_id}.pickle[.bz2]`` — original format before unification
    - ``{mol_id:08d}.pkl[.bz2]`` — short-lived zero-padded format

    Args:
        mol_id: Molecule identifier.
        is_compressed: Whether the file is bz2-compressed.

    Returns:
        List of legacy filenames to probe, in preference order.
    """
    suffix = ".bz2" if is_compressed else ""
    return [
        f"{int(mol_id):08d}.pkl{suffix}",
        f"{mol_id}.pickle{suffix}",
    ]


def load_frag_d(mol_id: str | int, frag_dp: str, is_compressed: bool = False) -> dict | None:
    """Load a fragment DAG dictionary from disk.

    Supports folder, tar archive, and zip archive backends.  The canonical
    filename format is ``{mol_id}.pkl[.bz2]`` (see :func:`get_frag_name`).
    For backward compatibility, the zero-padded ``{mol_id:08d}.pkl[.bz2]`` and
    the original ``{mol_id}.pickle[.bz2]`` formats are also tried when the
    canonical file is not found.

    Args:
        mol_id: Molecule identifier (int or numeric string).
        frag_dp: Directory path or archive file (``.tar`` / ``.zip``) containing
            fragment DAG files.
        is_compressed: Whether the files are bz2-compressed. Defaults to False.

    Returns:
        Loaded fragment dictionary, or ``None`` if not found inside a tar archive.
    """
    if os.path.isfile(frag_dp) and str(frag_dp).endswith(".tar"):
        canonical = get_frag_name(mol_id, is_compressed)
        accepted = {canonical} | set(_legacy_frag_names(mol_id, is_compressed))
        with tarfile.open(frag_dp, "r") as tar_read:
            for member in tar_read.getmembers():
                if member.name in accepted:
                    f = tar_read.extractfile(member)
                    if f is None:
                        raise ValueError(
                            f"Could not extract '{member.name}' from archive '{frag_dp}'"
                        )
                    content = f.read()
                    if member.name.endswith(".bz2"):
                        return pickle.loads(bz2.decompress(content))
                    return pickle.loads(content)
        return None

    if os.path.isfile(frag_dp) and str(frag_dp).endswith(".zip"):
        canonical = get_frag_name(mol_id, is_compressed)
        with ZipFile(frag_dp, "r") as zip_read:
            names = zip_read.namelist()
            member_name = canonical
            if member_name not in names:
                for legacy in _legacy_frag_names(mol_id, is_compressed):
                    if legacy in names:
                        member_name = legacy
                        break
            with zip_read.open(member_name) as f:
                content = f.read()
        if member_name.endswith(".bz2"):
            return pickle.loads(bz2.decompress(content))
        return pickle.loads(content)

    fp = get_frag_fp(mol_id, frag_dp, is_compressed)
    if not os.path.isfile(fp):
        for legacy in _legacy_frag_names(mol_id, is_compressed):
            legacy_fp = os.path.join(frag_dp, legacy)
            if os.path.isfile(legacy_fp):
                fp = legacy_fp
                break
    return _load_pickle_file(fp)
