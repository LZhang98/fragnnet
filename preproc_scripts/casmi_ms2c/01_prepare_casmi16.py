
import os
import pickle
import signal
import sys
import traceback
from threading import Event
import argparse
import os
import io

import tarfile
import bz2
import numpy as np

from tqdm import tqdm
import pandas as pd
from frag_gnn.utils.ms2c_utils import PubChemCandidateDB
from frag_gnn.utils.misc_utils import booltype, PPM
import frag_gnn.utils.formula_utils as formula_utils
import frag_gnn.utils.frag_utils as frag_utils
import frag_gnn.utils.data_utils as data_utils
from frag_gnn.utils.misc_utils import booltype
from frag_gnn.frag.compute_frags import (
	MAX_NUM_NODES
)

from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from copy import deepcopy
from frag_gnn.utils.data_utils import par_apply_series, par_apply_df_rows, seq_apply_series, seq_apply_df_rows

if __name__ == "__main__":

	RDLogger.DisableLog('rdApp.*') 

	casmi_cat1_df = pd.read_csv('data/misc/casmi2016/solutions_casmi2016_cat1.csv')
	casmi_cat23_df = pd.read_csv('data/misc/casmi2016/solutions_casmi2016_cat2and3.csv')
	casmi_sol_df = pd.concat([casmi_cat1_df,casmi_cat23_df])
	print(casmi_sol_df)