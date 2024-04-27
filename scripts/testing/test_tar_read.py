import os
import tarfile
import time
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import bz2
import pickle
import pandas as pd

mol_fp = "./data/proc/nist20mona23/mol_df.pkl"
df = pd.read_pickle(mol_fp)
print(df.head())
#data_dir = "/home/feiw/gitRepos/frag-gnn/data/frag/nist20_d4_h4_isoFalse/"
#dags_dir = os.path.join(data_dir,'dags')
#dag_filenames = [f for f in os.listdir(dags_dir)]


#tar_ball_fp = f"{data_dir}/sample.tar"
#if os.path.isfile(tar_ball_fp):
#    os.remove(tar_ball_fp)

#with tarfile.open(tar_ball_fp, "a") as tar_out:
#    for f in tqdm(dag_filenames):
#        tar_out.add(os.path.join(dags_dir,f), arcname=f)
    
#tar_ball_fp = "./test.tar"
#with tarfile.open(tar_ball_fp, "r") as tar:
#    for member in tar.getmembers():
#        print(member.name)
#        f=tar.extractfile(member)
#        content=f.read()
        #print(type(content))

#        dag_d = pickle.loads(bz2.decompress(content))
#        print(dag_d)
#        exit()
#        #for f in tqdm(dag_filenames):
        #    tar_out.add(os.path.join(dags_dir,f), arcname=f)

#fp = "data/ms2c/pubchem/frag_dp/d4_h4_isoFalse/ASWBNKHCZGQVJV.pickle.bz2"
#with bz2.BZ2File(fp, 'rb') as filein: 
#	frag_d = pickle.load(filein)
#	print(frag_d)
