import argparse
import os
import pandas as pd
import numpy as np
import json

from frag_gnn.utils.misc_utils import booltype, np_temp_seed

def make_splits(args):

    mol_df = pd.read_pickle(os.path.join(args.proc_dp,"mol_df.pkl"))
    spec_df = pd.read_pickle(os.path.join(args.proc_dp,"spec_df.pkl"))
    dag_stats_df = pd.read_pickle(
        os.path.join(args.frag_dp,f"{args.dag_filter_grouping}_stats_df.pkl")
    )

    def get_max_mz(peaks):
        return max([peak[0] for peak in peaks])

    print("> Spectrum filters")
    # filter spectra
    dsets = args.dsets
    prec_types = args.prec_types
    spec_masks = []
    spec_masks.append(spec_df["dset"].isin(dsets))
    spec_masks.append(spec_df["prec_type"].isin(prec_types))
    spec_masks.append(spec_df["peaks"].apply(get_max_mz) <= args.max_peak_mz)
    spec_masks = np.stack(spec_masks,axis=1)
    spec_mask = np.all(spec_masks,axis=1)
    print(np.sum(spec_mask), spec_mask.shape[0], np.mean(spec_mask))
    spec_df = spec_df[spec_mask]
    spec_spec_id = spec_df["spec_id"]
    spec_group_id = spec_df["group_id"]
    spec_mol_id = spec_df["mol_id"]

    print("> DAG filters")
    # filter dags
    node_key = "dag_num_nodes"
    wrecall_key = "wrecall_10ppm_h4"
    dag_masks = []
    if args.max_num_dag_nodes != -1:
        dag_masks.append(dag_stats_df[node_key] <= args.max_num_dag_nodes)
    dag_masks.append(dag_stats_df[wrecall_key] >= args.min_dag_wrecall)
    dag_masks = np.stack(dag_masks,axis=1)
    dag_mask = np.all(dag_masks,axis=1)
    print(np.sum(dag_mask), dag_mask.shape[0], np.mean(dag_mask))
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
    if args.dag_filter_grouping in ["m_spec","m_mol"]:
        both_group_id = np.intersect1d(spec_mol_id,dag_group_id)
        split_df = spec_df[spec_df["group_id"].isin(dag_group_id)]
    else:
        both_spec_id = np.intersect1d(spec_spec_id,dag_spec_id)
        split_df = spec_df[spec_df["spec_id"].isin(both_spec_id)]
    split_df = split_df.merge(mol_df[["mol_id",args.split_key]],on="mol_id",how="inner")

    if args.split_type == "random":
        # split based on molecule
        split_mol_id = np.unique(split_df["mol_id"])
        split_keys = split_df[split_df["mol_id"].isin(split_mol_id)][args.split_key]
        split_keys = np.unique(split_keys)
        total_num = split_keys.shape[0]
        test_num = int(np.ceil(total_num*args.test_frac))
        val_num = int(np.ceil(total_num*args.val_frac))
        train_num = total_num - test_num - val_num
        with np_temp_seed(args.meta_rseed):
            test_keys = np.random.choice(split_keys,size=test_num,replace=False)
            train_val_keys = np.setdiff1d(split_keys,test_keys)
            val_keys = np.random.choice(train_val_keys,size=val_num,replace=False)
            train_keys = np.setdiff1d(train_val_keys,val_keys)
        train_df = split_df[split_df[args.split_key].isin(train_keys)][["spec_id","mol_id","group_id"]]
        val_df = split_df[split_df[args.split_key].isin(val_keys)][["spec_id","mol_id","group_id"]]
        test_df = split_df[split_df[args.split_key].isin(test_keys)][["spec_id","mol_id","group_id"]]
    else:
        assert args.split_type == "predefined"
        assert len(args.train_ids) > 1 and len(args.val_ids) > 1 and len(args.test_ids) > 1
        train_ids = np.array([args.train_ids])
        val_ids = np.array(args.val_ids)
        test_ids = np.array(args.test_ids)
        train_df = split_df[split_df[args.id_type].isin(train_ids)][["spec_id","mol_id","group_id"]]
        val_df = split_df[split_df[args.id_type].isin(val_ids)][["spec_id","mol_id","group_id"]]
        test_df = split_df[split_df[args.id_type].isin(test_ids)][["spec_id","mol_id","group_id"]]

    # create split directory
    os.makedirs(args.split_dp,exist_ok=True)

    # save ids
    train_fp = os.path.join(args.split_dp,"train_ids.csv")
    val_fp = os.path.join(args.split_dp,"val_ids.csv")
    test_fp = os.path.join(args.split_dp,"test_ids.csv")
    train_df.to_csv(train_fp,index=False)
    val_df.to_csv(val_fp,index=False)
    test_df.to_csv(test_fp,index=False)

    # save split metadata
    meta_d = vars(args)
    meta_fp = os.path.join(args.split_dp,"meta.json")
    with open(meta_fp,"w") as f:
        json.dump(meta_d,f,indent=4)

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--split_type",type=str,choices=["random","predefined"],default="random")
    parser.add_argument("--split_key",type=str,choices=["inchikey_s","scaffold"],default="inchikey_s")
    parser.add_argument("--id_type",type=str,choices=["spec_id","mol_id","group_id"],default="group_id")
    parser.add_argument("--train_ids",type=int,nargs="+",default=[62611])
    parser.add_argument("--val_ids",type=int,nargs="+",default=[62611])
    parser.add_argument("--test_ids",type=int,nargs="+",default=[62611])
    # spec filtering criteria
    parser.add_argument("--dsets",type=str,nargs="+",default=["nist"])
    parser.add_argument("--prec_types",type=str,nargs="+",default=["[M+H]+"])
    parser.add_argument("--max_peak_mz",type=float,default=1500.)
    # dag filtering criteria
    parser.add_argument("--dag_filter_grouping",type=str,choices=["mol","spec","m_mol","m_spec"],default="m_spec")
    parser.add_argument("--max_num_dag_nodes",type=int,default=10000)
    parser.add_argument("--min_dag_wrecall",type=float,default=0.75)
    # non-filtering args
    parser.add_argument("--meta_rseed",type=int,default=420420)
    parser.add_argument("--test_frac",type=float,default=0.2)
    parser.add_argument("--val_frac",type=float,default=0.2)
    parser.add_argument("--proc_dp",type=str,default="data/proc/nist")
    parser.add_argument("--frag_dp",type=str,default="data/frag/d3_h4_isoTrue_v2")
    parser.add_argument("--split_dp",type=str,default="data/split/d3_h4_isoTrue_v2")

    args = parser.parse_args()

    make_splits(args)

