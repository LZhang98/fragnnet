import argparse
import bz2
import glob
import os
import pickle

import h5py
import numpy as np
import torch
from torch_geometric.data import Data as PyGData


def dag_to_feature_vector(dag_dict, hdf5_keys):
    """
    Convert a DAG dictionary to a fixed-length feature vector using the provided keys.
    For each key, tries to extract a scalar value:
      - If int/float: use as is
      - If list/array: use mean (or 0 if empty)
      - If torch.Tensor: use mean (or 0 if empty)
      - If dict: use length
      - If torch_geometric.data.Data: use number of nodes (len(v)), or 0 if not available
      - If None: 0
      - Otherwise: use length if possible, else 0
    """

    features_d = {}
    for k in dag_dict.keys():
        print(k)
        v = dag_dict.get(k)
        if isinstance(v, (int, float, np.number)):
            features_d[k] = v
        elif isinstance(v, list):
            print(k, v)
            features_d[k] = np.ndarray(v)
        elif isinstance(v, np.ndarray):
            features_d[k] = v
        elif isinstance(v, torch.Tensor):
            # Convert tensor to numpy and handle empty tensors
            if v.numel() == 0:
                features_d[k] = np.zeros(1, dtype=np.float32)
            else:
                features_d[k] = v.detach().cpu().numpy()
        elif v is None:
            features_d[k] = np.nan
        elif isinstance(v, PyGData):
            print(v)
            # Use number of nodes if possible, else 0.0
            # try:
            #   features.append(float(v.num_nodes))
            # except Exception:
            #   try:
            #       features.append(float(len(v)))
            #   except Exception:
            #       features.append(np.nan)
        else:
            print(f"Warning: Unhandled type for key '{k}': {type(v)}")
            # try:
            #   features.append(float(len(v)))
            # except Exception:
            #   features.append(0.0)
    exit()
    return np.array(features, dtype=np.float32)


def append_hdf5_rows(hdf5_fp, new_rows):
    """
    Append a batch of rows (new_rows) to the HDF5 dataset 'dags'.
    Opens the file in append mode, resizes the dataset, and writes the new data.
    """
    with h5py.File(hdf5_fp, "a") as f:
        dataset = f["dags"]
        current_size = dataset.shape[0]
        new_size = current_size + new_rows.shape[0]
        dataset.resize((new_size, new_rows.shape[1]))
        dataset[current_size:new_size, :] = new_rows
        f.flush()


def create_hdf5_file(hdf5_fp, num_features, batch_size):
    """
    Create a new HDF5 file with an extendable dataset 'dags'.
    The dataset is compressed and chunked for efficient I/O.
    """
    with h5py.File(hdf5_fp, "w") as f:
        f.create_dataset(
            "dags",
            shape=(0, num_features),
            maxshape=(None, num_features),
            dtype="f4",
            chunks=(batch_size, num_features),
            compression="gzip",
            compression_opts=4,
        )


if __name__ == "__main__":
    # Parse command-line arguments for input directory, output file, and batch size
    parser = argparse.ArgumentParser(
        description="Convert DAG pickle.bz2 files to a single HDF5 file."
    )
    parser.add_argument(
        "--dags_dp",
        type=str,
        default="data/frag/nist20_d3_h4_isoFalse_nb3/dags",
        help="Directory containing .pickle.bz2 DAG files",
    )
    parser.add_argument(
        "--hdf5_fp",
        type=str,
        default="data/frag/nist20_d3_h4_isoFalse_nb3.hdf5",
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for HDF5 chunking and writing"
    )
    args = parser.parse_args()

    dags_dp = args.dags_dp
    hdf5_fp = args.hdf5_fp
    batch_size = args.batch_size

    # List of keys for hdf5 dataset
    hdf5_keys = [
        "max_depth",
        "reached_depth",
        "h_prior",
        "max_h_transfer",
        "formula_peak_mzs",
        "formula_peak_probs",
        "idx_to_formula",
        "idx_by_h_delta",
        "dag_data_x",
        "dag_data_y",
        "dag_data_edge_index",
        "dag_data_edge_attr",
        "dag_data_featuer_sizeedges_min_depth",
        "nodes_min_depth",
        "dag_num_edges",
        "dag_num_nodes",
        "dag_num_nodes_nb",
        "dag_sparsity",
        "formula_redundancy",
        "node_feature_size",
        "edge_feature_size",
        "is_directed",
        "dag_num_edges_by_depth",
        "dag_num_nodes_by_depth",
    ]
    num_features = len(hdf5_keys)

    # Create the HDF5 file and dataset
    create_hdf5_file(hdf5_fp, num_features, batch_size)

    # Find all .pickle.bz2 files in the input directory
    dag_files = sorted(glob.glob(os.path.join(dags_dp, "*.pickle.bz2")))
    print(f"Found {len(dag_files)} DAG files.")

    # Process files in batches and write to HDF5
    batch = []
    for i, fp in enumerate(dag_files):
        with bz2.BZ2File(fp, "rb") as filein:
            dag_dict = pickle.load(filein)
        feat = dag_to_feature_vector(dag_dict, hdf5_keys)
        batch.append(feat)
        # Write batch to HDF5 when full or at the end
        if len(batch) == batch_size or i == len(dag_files) - 1:
            batch_np = np.stack(batch, axis=0)
            append_hdf5_rows(hdf5_fp, batch_np)
            print(f"Wrote batch {i // batch_size + 1} to HDF5.")
            batch = []

    # Print final dataset shape for verification
    with h5py.File(hdf5_fp, "r") as f:
        dataset = f["dags"]
        print("Final dataset shape:", dataset.shape)
