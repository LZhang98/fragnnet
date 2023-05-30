import argparse
import os
import pandas as pd
import numpy as np
import json

from frag_gnn.utils.misc_utils import booltype, np_temp_seed

def make_splits(args):

    if args.split_type == "predefined":
        raise NotImplementedError

    mol_df = pd.read_pickle(os.path.join(args.proc_dp,"mol_df.pkl"))
    spec_df = pd.read_pickle(os.path.join(args.proc_dp,"spec_df.pkl"))
    # with open(os.path.join(args.frag_dp,args.frag_name,"global_stats.json"),"r",encoding="utf-8") as f:
    #     dag_global_stats = json.load(f)
    dag_stats_df = pd.read_pickle(
        os.path.join(args.frag_dp,args.frag_name,f"{args.dag_filter_grouping}_stats_df.pkl")
    )

    def get_max_mz(peaks):
        return max([peak[0] for peak in peaks])

    print("> Spectrum filters")
    # filter spectra
    dsets = args.dsets.split("-")
    prec_types = args.prec_types.split("-")
    spec_masks = []
    spec_masks.append(spec_df["dset"].isin(dsets))
    spec_masks.append(spec_df["prec_type"].isin(prec_types))
    spec_masks.append(spec_df["peaks"].apply(get_max_mz) <= args.max_peak_mz)
    spec_mask = np.all(spec_masks,axis=0)
    spec_df = spec_df[spec_mask]
    spec_spec_id = spec_df["spec_id"]
    spec_group_id = spec_df["group_id"]
    spec_mol_id = spec_df["mol_id"]

    print("> DAG filters")
    # filter dags
    node_key = "dag_num_nodes"
    wrecall_key = "wrecall_10ppm_h4"
    dag_masks = []
    dag_masks.append(dag_stats_df[node_key] <= args.max_num_dag_nodes)
    dag_masks.append(dag_stats_df[wrecall_key] >= args.min_dag_wrecall)
    dag_mask = np.all(dag_masks,axis=0)
    dag_stats_df = dag_stats_df[dag_mask]
    dag_mol_id = dag_stats_df["mol_id"]
    if args.dag_filter_grouping in ["m_spec","m_mol"]:
        dag_spec_id = None
        dag_group_id = dag_stats_df["group_id"]
    else:
        dag_spec_id = dag_stats_df["spec_id"]
        dag_group_id = None

    print("> Intersection")
    # get intersection
    # TODO: fix this
    if args.dag_filter_grouping in ["m_spec","m_mol"]:
        both_group_id = np.intersect1d(spec_mol_id,dag_group_id)
        split_df = spec_df[spec_df["group_id"].isin(dag_group_id)]
    else:
        both_spec_id = np.intersect1d(spec_spec_id,dag_spec_id)
        split_df = spec_df[spec_df["spec_id"].isin(both_spec_id)]
    split_df = split_df.merge(mol_df[["mol_id",args.split_type]],on="mol_id",how="inner")

    # split based on molecule
    split_mol_id = np.unique(split_df["mol_id"])
    split_key = split_df[split_df["mol_id"].isin(split_mol_id)][args.split_type]
    split_key = np.unique(split_key)
    total_num = split_key.shape[0]
    test_num = int(np.ceil(total_num*args.test_frac))
    val_num = int(np.ceil(total_num*args.val_frac))
    train_num = total_num - test_num - val_num
    with np_temp_seed(args.meta_rseed):
        test_key = np.random.choice(split_key,size=test_num,replace=False)
        train_val_key = np.setdiff1d(split_key,test_key)
        val_key = np.random.choice(train_val_key,size=val_num,replace=False)
        train_key = np.setdiff1d(train_val_key,val_key)
    
    # create split directory
    os.makedirs(os.path.join(args.split_dp,args.split_name),exist_ok=True)

    # save ids
    train_df = split_df[split_df[args.split_type].isin(train_key)][["spec_id","mol_id","group_id"]]
    val_df = split_df[split_df[args.split_type].isin(val_key)][["spec_id","mol_id","group_id"]]
    test_df = split_df[split_df[args.split_type].isin(test_key)][["spec_id","mol_id","group_id"]]
    train_fp = os.path.join(args.split_dp,args.split_name,"train_ids.csv")
    val_fp = os.path.join(args.split_dp,args.split_name,"val_ids.csv")
    test_fp = os.path.join(args.split_dp,args.split_name,"test_ids.csv")
    train_df.to_csv(train_fp,index=False)
    val_df.to_csv(val_fp,index=False)
    test_df.to_csv(test_fp,index=False)

    # save split metadata
    meta_d = vars(args)
    meta_fp = os.path.join(args.split_dp,args.split_name,"meta.json")
    with open(meta_fp,"w") as f:
        json.dump(meta_d,f,indent=4)

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--split_type",type=str,choices=["inchikey_s","scaffold","predefined"],default="inchikey_s")
    # spec filtering criteria
    parser.add_argument("--dsets",type=str,default="nist")
    parser.add_argument("--prec_types",type=str,default="[M+H]+")
    parser.add_argument("--max_peak_mz",type=float,default=1500.)
    # dag filtering criteria
    parser.add_argument("--dag_filter_grouping",type=str,choices=["mol","spec","m_mol","m_spec"],default="m_spec")
    parser.add_argument("--max_num_dag_nodes",type=int,default=10000)
    parser.add_argument("--min_dag_wrecall",type=float,default=0.75)
    # non-filtering args
    parser.add_argument("--meta_rseed",type=int,default=420420)
    parser.add_argument("--test_frac",type=float,default=0.2)
    parser.add_argument("--val_frac",type=float,default=0.2)
    parser.add_argument("--proc_dp",type=str,default="data/proc")
    parser.add_argument("--frag_dp",type=str,default="data/frag")
    parser.add_argument("--split_dp",type=str,default="data/split")
    parser.add_argument("--split_name",type=str,default="debug")
    parser.add_argument("--frag_name",type=str,default="debug")

    args = parser.parse_args()

    make_splits(args)

