
import pandas as pd
fp = "./data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50/iceberg_inchikey_s0_sanity.pkl"
#fp = "./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50_sanity/spec_df.pkl.gz"
df = pd.read_pickle(fp)
print(df.shape)
#print(df["group_id"].nunique())
print(df['cos_sim_0.01'].mean())