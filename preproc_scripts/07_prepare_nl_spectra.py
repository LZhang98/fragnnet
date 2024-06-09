import sys
import argparse
import logging
import json
from pathlib import Path
import numpy as np
import copy
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from pprint import pformat
import torch as th
import yaml

import shutil

if __name__ == "__main__":
	spec_df_dp = './data/proc/pcdl/spec_df_org.pkl'
	spec_df = pd.read_pickle(spec_df_dp)
	#print(spec_df)
	
	reverse_spec_df = spec_df.copy()
	reverse_spec_df["prec_type"] = reverse_spec_df["prec_type"] + "_NL"
	reverse_spec_df["peaks"] = reverse_spec_df.apply(lambda x : [(x['prec_mz']-i,j) for i,j in x["peaks"] if abs(x['prec_mz']-i) > 1], axis =1)
	reverse_spec_df = reverse_spec_df[reverse_spec_df['peaks'].map(lambda x: len(x)) > 0]

	extended_spectra_df = pd.concat([spec_df, reverse_spec_df])
	extended_spectra_df.reset_index(inplace=True,drop=True)
	extended_spectra_df['spec_id'] = extended_spectra_df.index
	extended_spectra_df.to_pickle('./data/proc/pcdl/spec_df.pkl')