import pandas as pd
import os

v2_dp = "data/ms2c/pubchem/predicted/v2/inchikey_s0_fraggnn_d4_v2_pubchem_nist20_d4_wr00_inchikey_test"
fp = os.path.join(v2_dp,"15.pkl.gz")
df = pd.read_pickle(fp)
print(df.columns)