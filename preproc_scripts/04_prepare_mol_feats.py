import pandas as pd
import argparse

from frag_gnn.feat_utils import MolGraph
from frag_gnn.data_utils import make_morgan_fingerprint, make_maccs_fingerprint, make_rdkit_fingerprint

def main(args):

    # read in molecule df
    mol_df = pd.read_pickle(os.path.join(args.project_dp,args.proc_dp,"mol_df.pkl"))

    # prepare fingerprint features
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
