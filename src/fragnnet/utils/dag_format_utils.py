"""Fragment-DAG storage format: file naming, the pickle and packed-HDF5 writers, and the readers.

Core code (``base_dataset``) reads DAGs, so the format lives in core, with reader and writer in one
place (docs/plans/repo_reorg_plan.md, "frag_utils is split, not moved"). FraGNNet's DAG builder in
``frag_utils`` writes through it.
"""

import bz2
import gzip
import json
import os
import pickle
import tarfile
import zlib
from typing import TYPE_CHECKING
from zipfile import ZipFile

import numpy as np
import torch as th
import torch_geometric as pyg

if TYPE_CHECKING:
    import h5py as _h5py

PACKED_DAG_FORMAT_ATTR = "fragnnet_dag_format"
# Bump whenever PACKED_DAG_INDEX_DTYPE's on-disk layout changes, so
# initialize_packed_dag_hdf5 rejects an old file instead of silently
# reusing an incompatible index dtype.
PACKED_DAG_FORMAT = "packed_v1"
PACKED_DAG_BYTES_DATASET = "dag_bytes"
PACKED_DAG_INDEX_DATASET = "dag_index"
PACKED_DAG_BYTES_CHUNK_SIZE = 64 * 1024
PACKED_DAG_INDEX_CHUNK_SIZE = 8192
# zlib level for the per-molecule DAG payload. Measured on 185 drug-like d3/h4 DAGs
# (150 MB of raw pickle): level 4 -> 13.5 MB, level 6 -> 12.2 MB, level 9 -> 12.0 MB.
# Level 6 is strictly better than 4 here — 9.5% smaller AND ~10% faster to inflate
# (0.83 vs 0.92 ms/DAG, since there is less to read back) for +3 ms/molecule of write
# CPU, against ~90 ms/molecule of fragmentation. Level 9 costs 20x the write CPU for a
# further 1.5%. No HDF5-level filter is applied on top: the bytes dataset already holds
# compressed payloads, so a gzip filter would mostly duplicate this work.
PACKED_DAG_ZLIB_LEVEL = 6
PACKED_DAG_INDEX_DTYPE = np.dtype(
    [
        ("offset", "<u8"),
        ("length", "<u8"),
    ]
)


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


def initialize_packed_dag_hdf5(h5_file: "_h5py.File") -> None:
    """Initialize an open HDF5 file for packed DAG storage.

    Creates the shared byte-blob and index datasets used by
    :func:`dump_dag_hdf5_packed`. Idempotent: calling this again on a file
    already initialized for packed storage is a no-op.

    Args:
        h5_file: Open writable ``h5py.File`` object.

    Raises:
        ValueError: If the file already stores DAGs in an unsupported packed
            format, or if the file is non-empty but not yet initialized for
            packed DAG storage.
    """
    stored_format = h5_file.attrs.get(PACKED_DAG_FORMAT_ATTR)
    if stored_format is not None:
        if stored_format != PACKED_DAG_FORMAT:
            raise ValueError(f"Unsupported packed DAG format: {stored_format}")
        return
    if len(h5_file) != 0:
        raise ValueError("Cannot initialize packed DAG storage in a non-empty HDF5 file")

    h5_file.attrs[PACKED_DAG_FORMAT_ATTR] = PACKED_DAG_FORMAT
    h5_file.create_dataset(
        PACKED_DAG_BYTES_DATASET,
        shape=(0,),
        maxshape=(None,),
        chunks=(PACKED_DAG_BYTES_CHUNK_SIZE,),
        dtype=np.uint8,
    )
    h5_file.create_dataset(
        PACKED_DAG_INDEX_DATASET,
        shape=(0,),
        maxshape=(None,),
        chunks=(PACKED_DAG_INDEX_CHUNK_SIZE,),
        dtype=PACKED_DAG_INDEX_DTYPE,
    )


def dump_dag_hdf5_packed(h5_file: "_h5py.File", mol_id: str | int, dag_d: dict) -> None:
    """Append one compressed DAG payload to a packed HDF5 file.

    Args:
        h5_file: Open writable ``h5py.File``, already initialized via
            :func:`initialize_packed_dag_hdf5`.
        mol_id: Molecule identifier. Used directly as a dense index into the
            packed index dataset, so it must be a small non-negative integer.
        dag_d: DAG payload dictionary as returned by :func:`compute_dags`.

    Raises:
        ValueError: If ``mol_id`` is negative, or a DAG has already been
            stored for it.
        KeyError: If ``h5_file`` was not initialized via
            :func:`initialize_packed_dag_hdf5`.
    """
    mol_idx = int(mol_id)
    if mol_idx < 0:
        raise ValueError(f"mol_id must be non-negative for packed HDF5 storage: {mol_id}")

    index_dataset = h5_file[PACKED_DAG_INDEX_DATASET]
    if mol_idx < index_dataset.shape[0] and int(index_dataset[mol_idx]["length"]) != 0:
        raise ValueError(f"Duplicate mol_id in packed HDF5 file: {mol_id}")

    payload = pickle.dumps(dag_d, protocol=pickle.HIGHEST_PROTOCOL)
    compressed_payload = zlib.compress(payload, level=PACKED_DAG_ZLIB_LEVEL)
    payload_array = np.frombuffer(compressed_payload, dtype=np.uint8)

    bytes_dataset = h5_file[PACKED_DAG_BYTES_DATASET]
    offset = int(bytes_dataset.shape[0])
    length = int(payload_array.size)
    bytes_dataset.resize((offset + length,))
    bytes_dataset[offset : offset + length] = payload_array

    if mol_idx >= index_dataset.shape[0]:
        index_dataset.resize((mol_idx + 1,))
    index_dataset[mol_idx] = (offset, length)


def _load_dag_hdf5_packed(h5_file, mol_id: str | int) -> dict:
    """Load one DAG payload from a packed HDF5 file."""
    mol_idx = int(mol_id)
    index_dataset = h5_file[PACKED_DAG_INDEX_DATASET]
    if mol_idx < 0 or mol_idx >= index_dataset.shape[0]:
        raise KeyError(f"mol_id {mol_id} is not present in packed HDF5 file")

    record = index_dataset[mol_idx]
    length = int(record["length"])
    if length == 0:
        raise KeyError(f"mol_id {mol_id} is not present in packed HDF5 file")

    offset = int(record["offset"])
    compressed_payload = h5_file[PACKED_DAG_BYTES_DATASET][offset : offset + length].tobytes()
    return pickle.loads(zlib.decompress(compressed_payload))


def load_dag_hdf5(h5_file: "_h5py.File", mol_id: str | int) -> dict:
    """Read one DAG dictionary from an open HDF5 file, packed or legacy format.

    Args:
        h5_file: Open h5py.File in the packed format (:func:`dump_dag_hdf5_packed`) or the
            legacy one-group-per-molecule format.
        mol_id: Molecule identifier to look up.

    Returns:
        DAG payload dictionary compatible with the format returned by
        ``compute_dags``.
    """
    if h5_file.attrs.get(PACKED_DAG_FORMAT_ATTR) == PACKED_DAG_FORMAT:
        return _load_dag_hdf5_packed(h5_file, mol_id)
    return load_dag_hdf5_from_group(h5_file[f"{int(mol_id)}"])


def load_dag_hdf5_from_group(h5_group: "_h5py.Group") -> dict:
    """Read one DAG dictionary from a legacy per-molecule HDF5 group.

    Numpy arrays are converted back to PyTorch tensors with their original
    dtypes.  JSON-encoded attributes are decoded to their original Python types.

    Args:
        h5_group: An open legacy molecule group (one group per molecule, the layout before
            the packed format).

    Returns:
        DAG payload dictionary compatible with the format returned by
        ``compute_dags``.
    """
    g = h5_group["dag"]
    dag = pyg.data.Data(
        x=th.from_numpy(g["x"][:]),
        edge_index=th.from_numpy(g["edge_index"][:]),
        edge_attr=th.from_numpy(g["edge_attr"][:]),
    )
    dag.node_feat_idxs = th.from_numpy(g["node_feat_idxs"][:])
    dag.edge_feat_idxs = th.from_numpy(g["edge_feat_idxs"][:])
    for name in ("boundary_pair_frag_idxs", "boundary_pair_in_local", "boundary_pair_out_local"):
        if name in g:
            setattr(dag, name, th.from_numpy(g[name][:]))

    dag_d: dict = {
        "dag": dag,
        "formula_peak_mzs": th.from_numpy(h5_group["formula_peak_mzs"][:]),
        "formula_peak_probs": th.from_numpy(h5_group["formula_peak_probs"][:]),
        "edges_min_depth": h5_group["edges_min_depth"][:],
        "nodes_min_depth": h5_group["nodes_min_depth"][:],
    }

    # JSON-encoded variable-structure fields
    dag_d["idx_to_formula"] = {
        int(k): v for k, v in json.loads(h5_group.attrs["idx_to_formula"]).items()
    }
    # Stored as list-of-lists; restore as list-of-sets to match original type
    dag_d["idx_by_h_delta"] = [set(s) for s in json.loads(h5_group.attrs["idx_by_h_delta"])]
    dag_d["dag_num_edges_by_depth"] = {
        int(k): v for k, v in json.loads(h5_group.attrs["dag_num_edges_by_depth"]).items()
    }
    dag_d["dag_num_nodes_by_depth"] = {
        int(k): v for k, v in json.loads(h5_group.attrs["dag_num_nodes_by_depth"]).items()
    }

    # Integer scalars
    for key in [
        "max_depth",
        "reached_depth",
        "max_h_transfer",
        "dag_num_edges",
        "dag_num_nodes",
        "dag_num_nodes_nb",
        "node_feature_size",
        "edge_feature_size",
    ]:
        dag_d[key] = int(h5_group.attrs[key])

    # Float scalars
    for key in ["dag_sparsity", "formula_redundancy"]:
        dag_d[key] = float(h5_group.attrs[key])

    # Bool scalars
    for key in ["h_prior", "is_directed", "force_stopped"]:
        dag_d[key] = bool(h5_group.attrs[key])

    return dag_d


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
