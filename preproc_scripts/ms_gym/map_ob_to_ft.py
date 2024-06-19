import pandas as pd
import os

ms_gym_spec_df = pd.read_pickle("data/proc/ms_gym/spec_df.pkl")
print(ms_gym_spec_df["inst_type"].unique())
ms_gym_spec_df.loc[ms_gym_spec_df["inst_type"] == 'Orbitrap',"inst_type"] = "FT"
print(ms_gym_spec_df["inst_type"].unique())
ms_gym_spec_df.to_pickle("data/proc/ms_gym/spec_df.pkl")