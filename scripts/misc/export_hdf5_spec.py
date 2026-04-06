# post process predicted spectra
import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py


def not_nan_none(val):
    return val is not None and not (isinstance(val, float) and np.isnan(val))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proc_dp", default="data/proc/nps_nist23", type=str, help="Processed data directory"
    )
    parser.add_argument(
        "--hdf5_path", default="data/proc/nps_nist23", type=str, help="pred hdf5 path"
    )
    parser.add_argument(
        "--output_fp", default="nps_nist23.mgf", type=str, help="Destination pickle file path"
    )

    args = parser.parse_args()
    proc_dp = args.proc_dp
    mgf_fp = args.output_fp
    pred_file=h5py.File(args.hdf5_path)

    print("Loading reference data...")
    mol_df_path_pkl = os.path.join(proc_dp, "mol_df.pkl")
    mol_df = pd.read_pickle(mol_df_path_pkl)
    print(f"Loaded mol_df with {len(mol_df)} molecules from {mol_df_path_pkl}")

    spec_df_path_pkl = os.path.join(proc_dp, "spec_df.pkl")
    spec_df = pd.read_pickle(spec_df_path_pkl)
    print(f"Loaded spec_df with {len(spec_df)} spectra from {spec_df_path_pkl}")

    merged_df = spec_df.merge(
        mol_df, how="left", left_on="mol_id", right_on="mol_id"
    )  # , suffixes=('_spec', '_mol'))
    print(f"Merged spec_df and mol_df into merged_df with {len(merged_df)} entries")
    # print(merged_df.columns.tolist())
    # Ensure output directories exist
    if os.path.dirname(mgf_fp) != "":
        os.makedirs(os.path.dirname(mgf_fp), exist_ok=True)
    map_spec_id_to_smi={}
    for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
        map_spec_id_to_smi[row['spec_id']]=[row["smiles"], row["inchikey_s"], row["prec_type"], row["nce"], row["ace"], row['formula'],  row["dset"], row["prec_mz"]]
    all_spec_id=np.array(pred_file["spec_id"][:], dtype=np.int64)
    start_points=np.array(pred_file["start_points"][:], dtype=np.int64)
    end_points=np.array(pred_file["end_points"][:], dtype=np.int64)#pred_file["end_points"][:]
    all_mz=pred_file["pred_mzs"]
    all_ints=pred_file["pred_ints"]
    num_pred_specs=len(all_spec_id)
    with open(mgf_fp, "w") as mgf_file:
        for idx in tqdm(range(num_pred_specs)):
            spec_id=all_spec_id[idx]
            s, e=start_points[idx], end_points[idx]
            mgf_file.write("BEGIN IONS\n")
            mgf_file.write(f"TITLE={spec_id}\n")
            smi, inchi, prec_type, nce, ace, formula,dset, prec_mz=map_spec_id_to_smi[spec_id]
            
            mgf_file.write(f"SMILES={smi}\n")
            mgf_file.write(f"INCHIKEY_S={inchi}\n")
            mgf_file.write(f"PRECURSORTYPE={prec_type}\n")
            mgf_file.write(f"COLLISIONENERGY={nce}% NCE\n")
            mgf_file.write(f"COLLISIONENERGY={ace}eV\n")
            mgf_file.write(f"FORMULA={formula}\n")
            mgf_file.write(f"DATASET={dset}\n")
            mgf_file.write(f"PEPMASS={prec_mz}\n")

            # Write peaks, filter out intensity < 1
            peaks = np.stack([all_mz[s:e], all_ints[s:e]], -1)
            for mz, intensity in peaks:
                mgf_file.write(f"{mz:.6f} {intensity:.1f}\n")
            mgf_file.write("END IONS\n\n")
