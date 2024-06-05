import pandas as pd
import argparse
import os
import pandas as pd
import numpy as np
import json

from frag_gnn.utils import frag_utils
from frag_gnn.utils.misc_utils import booltype, np_temp_seed
from frag_gnn.utils.frag_utils import ELEMENTS
from frag_gnn.utils.proc_utils import filter_spec_mol
import shutil
from tqdm import tqdm
import os

ms_gym_spec_df = pd.read_pickle("data/proc/ms_gym/spec_df.pkl")
print(ms_gym_spec_df["inst_type"].unique())
ms_gym_spec_df.loc[ms_gym_spec_df["inst_type"] == 'Orbitrap',"inst_type"] = "FT"
print(ms_gym_spec_df["inst_type"].unique())
ms_gym_spec_df.to_pickle("data/proc/ms_gym/spec_df.pkl")