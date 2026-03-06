"""Tests for fragment pickle IO helpers in ``fragnnet.utils.frag_utils``."""

import bz2
import io
import pickle
import tarfile
import zipfile

from fragnnet.utils import frag_utils


def test_save_and_load_frag_local_uncompressed(tmp_path):
    """Round-trip fragment payload using local uncompressed pickle file."""
    payload = {"dag": {"num_nodes": 3}, "meta": [1, 2, 3]}
    mol_id = "123"

    frag_utils.save_frag_d(payload, mol_id, str(tmp_path), is_compressed=False)
    loaded = frag_utils.load_frag_d(mol_id, str(tmp_path), is_compressed=False)

    assert loaded == payload


def test_save_and_load_frag_local_bz2(tmp_path):
    """Round-trip fragment payload using local bz2-compressed pickle file."""
    payload = {"dag": {"num_nodes": 5}, "meta": [4, 5, 6]}
    mol_id = "456"

    frag_utils.save_frag_d(payload, mol_id, str(tmp_path), is_compressed=True)
    loaded = frag_utils.load_frag_d(mol_id, str(tmp_path), is_compressed=True)

    assert loaded == payload


def test_load_frag_from_tar_uncompressed_member(tmp_path):
    """Load fragment from an uncompressed member inside a tar archive."""
    payload = {"dag": {"num_nodes": 7}, "meta": [7, 8, 9]}
    mol_id = "789"
    member_name = frag_utils.get_frag_name(mol_id, is_compressed=False)
    tar_fp = tmp_path / "frags.tar"

    content = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    info = tarfile.TarInfo(name=member_name)
    info.size = len(content)
    with tarfile.open(tar_fp, "w") as tf:
        tf.addfile(info, io.BytesIO(content))

    loaded = frag_utils.load_frag_d(mol_id, str(tar_fp), is_compressed=False)
    assert loaded == payload


def test_load_frag_from_tar_bz2_member(tmp_path):
    """Load fragment from a bz2-compressed member inside a tar archive."""
    payload = {"dag": {"num_nodes": 9}, "meta": [10, 11, 12]}
    mol_id = "321"
    member_name = frag_utils.get_frag_name(mol_id, is_compressed=True)
    tar_fp = tmp_path / "frags_bz2.tar"

    pickled = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    content = bz2.compress(pickled)
    info = tarfile.TarInfo(name=member_name)
    info.size = len(content)
    with tarfile.open(tar_fp, "w") as tf:
        tf.addfile(info, io.BytesIO(content))

    loaded = frag_utils.load_frag_d(mol_id, str(tar_fp), is_compressed=True)
    assert loaded == payload


def test_load_frag_from_zip_uncompressed_member(tmp_path):
    """Load fragment from an uncompressed member inside a zip archive."""
    payload = {"dag": {"num_nodes": 11}, "meta": [13, 14, 15]}
    mol_id = "654"
    member_name = frag_utils.get_frag_name(mol_id, is_compressed=False)
    zip_fp = tmp_path / "frags.zip"

    content = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    with zipfile.ZipFile(zip_fp, "w") as zf:
        zf.writestr(member_name, content)

    loaded = frag_utils.load_frag_d(mol_id, str(zip_fp), is_compressed=False)
    assert loaded == payload


def test_load_frag_from_zip_bz2_member(tmp_path):
    """Load fragment from a bz2-compressed member inside a zip archive."""
    payload = {"dag": {"num_nodes": 13}, "meta": [16, 17, 18]}
    mol_id = "987"
    member_name = frag_utils.get_frag_name(mol_id, is_compressed=True)
    zip_fp = tmp_path / "frags_bz2.zip"

    pickled = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    content = bz2.compress(pickled)
    with zipfile.ZipFile(zip_fp, "w") as zf:
        zf.writestr(member_name, content)

    loaded = frag_utils.load_frag_d(mol_id, str(zip_fp), is_compressed=True)
    assert loaded == payload
