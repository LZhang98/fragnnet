import pandas as pd

from frag_gnn.utils.proc_utils import filter_spec_mol
from frag_gnn.utils.frag_utils import ELEMENTS

# check how many mols/specs are dropped
# check the inchikey/scaffold splits are not super messed up
# check that the entries that remain are the same
# look at cases where the mols for some dropped spectra are still in the db

if __name__ == "__main__":

    do_mona = True

    # original preprocessed data files
    mol_df = pd.read_pickle("data/proc/nist20mona23/mol_df.pkl")
    spec_df = pd.read_pickle("data/proc/nist20mona23/spec_df.pkl")

    # original split files
    inchikey_train_ids = pd.read_csv("data/split/nist20_mona23_d4_wr00_inchikey/train_ids.csv",dtype=int)
    inchikey_val_ids = pd.read_csv("data/split/nist20_mona23_d4_wr00_inchikey/val_ids.csv",dtype=int)
    inchikey_test_ids = pd.read_csv("data/split/nist20_mona23_d4_wr00_inchikey/test_ids.csv",dtype=int)
    inchikey_sec_ids = pd.read_csv("data/split/nist20_mona23_d4_wr00_inchikey/secondary_ids.csv",dtype=int)
    scaffold_train_ids = pd.read_csv("data/split/nist20_mona23_d4_wr00_scaffold/train_ids.csv",dtype=int)
    scaffold_val_ids = pd.read_csv("data/split/nist20_mona23_d4_wr00_scaffold/val_ids.csv",dtype=int)
    scaffold_test_ids = pd.read_csv("data/split/nist20_mona23_d4_wr00_scaffold/test_ids.csv",dtype=int)
    scaffold_sec_ids = pd.read_csv("data/split/nist20_mona23_d4_wr00_scaffold/secondary_ids.csv",dtype=int)

    # new preprocessed data files
    if do_mona:
        mol_df_2 = pd.read_pickle("data/proc/debug_annot_6/mol_df.pkl")
        spec_df_2 = pd.read_pickle("data/proc/debug_annot_6/spec_df.pkl")
        ann_df_2 = pd.read_pickle("data/proc/debug_annot_6/ann_df.pkl")
    else:
        mol_df_2 = pd.read_pickle("data/proc/debug_annot_5/mol_df.pkl")
        spec_df_2 = pd.read_pickle("data/proc/debug_annot_5/spec_df.pkl")
        ann_df_2 = pd.read_pickle("data/proc/debug_annot_5/ann_df.pkl")

    # get map from old spec_ids to new spec_ids
    id_map_df = spec_df[["dset","dset_spec_id","spec_id"]].merge(spec_df_2[["dset","dset_spec_id","spec_id"]], on=["dset","dset_spec_id"], how="inner")

    spec_df_both = spec_df.merge(spec_df_2, on=["dset","dset_spec_id"], how="inner")
    for flag in ["x","y"]:
        spec_df_both = spec_df_both.drop(columns=[f"spec_id_{flag}",f"mol_id_{flag}",f"group_id_{flag}"])
    print(spec_df_both.columns)
    for column in spec_df.columns:
        if column not in ["dset","dset_spec_id","spec_id","mol_id","group_id"]:
            print(f">>> column {column}")
            print(spec_df_both[f"{column}_x"].isna().sum(), spec_df_both[f"{column}_y"].isna().sum())
            print(((spec_df_both[f"{column}_x"] == spec_df_both[f"{column}_y"]) | (spec_df_both[f"{column}_x"].isna() & spec_df_both[f"{column}_y"].isna())).all())

    if do_mona:
        dsets = ["nist20_hr","mona23"]
    else:
        dsets = ["nist20_hr"]

    # filter
    f_spec_df, f_mol_df = filter_spec_mol(
        spec_df,
        mol_df,
        allowed_elements=ELEMENTS,
        dsets=dsets,
        prec_types=["[M+H]+"],
        max_peak_mz=1500.,
        num_entries=-1
    )
    
    # filter
    f_spec_df_2, f_mol_df_2 = filter_spec_mol(
        spec_df_2,
        mol_df_2,
        allowed_elements=ELEMENTS,
        dsets=dsets,
        prec_types=["[M+H]+"],
        max_peak_mz=1500.,
        num_entries=-1
    )

    print(f_mol_df.shape)
    print(f_spec_df.shape)

    print(f_mol_df_2.shape)
    print(f_spec_df_2.shape)

    # see how many dropped spectra still have molecules in the db
    lost_dset_spec_ids = f_spec_df[["dset","dset_spec_id"]].merge(f_spec_df_2[["dset","dset_spec_id"]], on=["dset","dset_spec_id"], how="inner").drop_duplicates()
    lost_mol_ids = f_spec_df.merge(lost_dset_spec_ids,how="inner")["mol_id"].unique()
    lost_smileses = f_mol_df[f_mol_df["mol_id"].isin(lost_mol_ids)]["smiles"].unique()
    print(f_mol_df_2["smiles"].isin(lost_smileses).sum())

    # see how much the splits change
    inchikey_train_ids_2 = id_map_df[id_map_df["spec_id_x"].isin(inchikey_train_ids["spec_id"])]["spec_id_y"]
    inchikey_train_ids_2 = f_spec_df_2[f_spec_df_2["spec_id"].isin(inchikey_train_ids_2)][["spec_id","mol_id","group_id"]]
    inchikey_val_ids_2 = id_map_df[id_map_df["spec_id_x"].isin(inchikey_val_ids["spec_id"])]["spec_id_y"]
    inchikey_val_ids_2 = f_spec_df_2[f_spec_df_2["spec_id"].isin(inchikey_val_ids_2)][["spec_id","mol_id","group_id"]]
    inchikey_test_ids_2 = id_map_df[id_map_df["spec_id_x"].isin(inchikey_test_ids["spec_id"])]["spec_id_y"]
    inchikey_test_ids_2 = f_spec_df_2[f_spec_df_2["spec_id"].isin(inchikey_test_ids_2)][["spec_id","mol_id","group_id"]]
    inchikey_sec_ids_2 = id_map_df[id_map_df["spec_id_x"].isin(inchikey_sec_ids["spec_id"])]["spec_id_y"]
    inchikey_sec_ids_2 = f_spec_df_2[f_spec_df_2["spec_id"].isin(inchikey_sec_ids_2)][["spec_id","mol_id","group_id"]]
    scaffold_train_ids_2 = id_map_df[id_map_df["spec_id_x"].isin(scaffold_train_ids["spec_id"])]["spec_id_y"]
    scaffold_train_ids_2 = f_spec_df_2[f_spec_df_2["spec_id"].isin(scaffold_train_ids_2)][["spec_id","mol_id","group_id"]]
    scaffold_val_ids_2 = id_map_df[id_map_df["spec_id_x"].isin(scaffold_val_ids["spec_id"])]["spec_id_y"]
    scaffold_val_ids_2 = f_spec_df_2[f_spec_df_2["spec_id"].isin(scaffold_val_ids_2)][["spec_id","mol_id","group_id"]]
    scaffold_test_ids_2 = id_map_df[id_map_df["spec_id_x"].isin(scaffold_test_ids["spec_id"])]["spec_id_y"]
    scaffold_test_ids_2 = f_spec_df_2[f_spec_df_2["spec_id"].isin(scaffold_test_ids_2)][["spec_id","mol_id","group_id"]]
    scaffold_sec_ids_2 = id_map_df[id_map_df["spec_id_x"].isin(scaffold_sec_ids["spec_id"])]["spec_id_y"]
    scaffold_sec_ids_2 = f_spec_df_2[f_spec_df_2["spec_id"].isin(scaffold_sec_ids_2)][["spec_id","mol_id","group_id"]]

    print(inchikey_train_ids.shape, inchikey_train_ids_2.shape)
    print(inchikey_val_ids.shape, inchikey_val_ids_2.shape)
    print(inchikey_test_ids.shape, inchikey_test_ids_2.shape)
    print(inchikey_sec_ids.shape, inchikey_sec_ids_2.shape)
    print(scaffold_train_ids.shape, scaffold_train_ids_2.shape)
    print(scaffold_val_ids.shape, scaffold_val_ids_2.shape)
    print(scaffold_test_ids.shape, scaffold_test_ids_2.shape)
    print(scaffold_sec_ids.shape, scaffold_sec_ids_2.shape)

    import pdb; pdb.set_trace()