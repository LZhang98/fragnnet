import pandas as pd
import torch as th
from torch.utils.data import Dataset
import numpy as np
import torch_geometric as pyg
from torch_geometric.data import Data, Batch

from frag_gnn.utils.feat_utils import batch_mols_frags, get_mol_fp, get_mol_graph, get_frag_graph
from frag_gnn.utils.spec_utils import thresh_func, filter_func, batch_func, prec_type_idx, transform_ce
from frag_gnn.utils.dataset_utils import merge_spec_df


def SpecMolDataset(Dataset):

    def __init__(self,mol_df,spec_df,split_df,mol_params,spec_params):

        super().__init__()
        self.mol_df = mol_df
        self.spec_df = spec_df
        self.split_df = split_df
        self.mol_params = mol_params
        self.spec_params = spec_params

        # select spectra
        self.spec_df = self.spec_df[self.spec_df["spec_id"].isin(self.split_df["spec_id"])]
        # select molecules
        self.mol_df = self.mol_df[self.mol_df["mol_id"].isin(self.split_df["mol_id"])]
        assert np.all(np.unique(self.mol_df["mol_id"]) == np.unique(self.spec_df["mol_id"]))

        # merge spectra
        self.merge = spec_params["merge"]
        if self.merge:
            self.spec_df = merge_spec_df(self.spec_df)            

        # subsample
        if self.spec_params["subsample_size"] > 0:
            self.spec_df = self.spec_df.sample(
                n=self.spec_params["subsample_size"],
                random_state=self.spec_params["subsample_seed"],
                replace=False)
            self.mol_df = self.mol_df[self.mol_df["mol_id"].isin(self.spec_df["mol_id"])]

        # reset indices
        self.spec_df = self.spec_df.reset_index(drop=True)
        self.mol_df = self.mol_df.reset_index(drop=True)

        # use mol_id as index for speedy access
        self.mol_df = self.mol_df.set_index("mol_id",drop=False).sort_index().rename_axis(None)

    def __len__(self):

        return len(self.spec_df)

    def __getitem__(self,idx):

        spec_entry = self.spec_df.iloc[idx]
        mol_id = spec_entry["mol_id"]
        mol_entry = self.mol_df.loc[mol_id]
        spec_data = self._process_spec(spec_entry)
        mol_data = self._process_mol(mol_entry)
        data = {**spec_data,**mol_data}
        return data

    def _process_spec(self,spec_entry):

        spec_data = {}
        # peak data
        peaks = spec_entry["peaks"]
        mzs, ints = [], []
        for peak in peaks:
            p_mz, p_int = peak
            mzs.append(p_mz)
            ints.append(p_int)
        mzs = th.tensor(mzs,dtype=th.float)
        ints = th.tensor(ints,dtype=th.float)
        mzs, ints = filter_func(mzs,ints,spec_params["ints_thresh"],spec_params["mz_max"])
        if self.spec_params["sparse"]:
            # get sparse spectrum
            spec_data["spec_mzs"] = mzs
            spec_data["spec_ints"] = ints
        if self.spec_params["binned"]:
            # bin spectrum
            spec_bin = bin_func(
                mzs,
                ints,
                spec_params["mz_max"],
                spec_params["mz_bin_res"],
                False,
                spec_params["sum_ints"]
            )
            spec_data["spec_bin"] = spec_bin.reshape(1,-1)
        # metadata
        if self.spec_params["prec_type"]:
            prec_type = spec_entry["prec_type"]
            prec_type = th.tensor([prec_type_idx(prec_type)],dtype=th.long)
            spec_data["spec_prec_type"] = prec_type
        if self.spec_params["prec_mass_diff"]:
            prec_type = spec_entry["prec_type"]
            mass_diff = th.tensor([prec_type_mass_diff(prec_type)],dtype=th.float)
            spec_data["spec_prec_mass_diff"] = mass_diff
        if self.spec_params["col_energy"]:
            assert not self.merged
            col_energy = spec_entry["col_energy"]
            col_energy = th.tensor([ce_transform(col_energy)],dtype=th.float)
            spec_data["spec_col_energy"] = col_energy
        if self.spec_params["prec_mz"]:
            prec_mz = spec_entry["prec_mz"]
            prec_mz = th.tensor([float(prec_mz)],dtype=th.float)
            spec_data["spec_prec_mz"] = prec_mz
        return spec_data

    def _process_mol(self,mol_entry):

        mol_data = {}
        mol = mol_entry["mol"]
        if self.mol_params["smiles"]:
            smiles = mol_entry["smiles"]
            mol_data["mol_smiles"] = [smiles]
        if self.mol_params["fp"]:
            fp = get_mol_fp(mol,
                self.mol_params["morgan"],
                self.mol_params["rdkit"],
                self.mol_params["maccs"]
            )
            mol_data["mol_fp"] = fp
        if self.mol_params["pyg"]:
            mol_pyg = get_mol_graph(
                mol,
                self.mol_params["atom_feats"],
                self.mol_params["bond_feats"],
                self.mol_params["pe_embed_k"],
                self.mol_params["bigraph"]
            )
            mol_data["mol_pyg"] = mol_pyg
        return mol_data

    @staticmethod
    def get_collate_fn():
        return SpecMolDataset.collate_fn

    @staticmethod
    def collate_fn(data_list):

        batch_size = len(data_list)
        keys = list(data_list[0].keys())
        collate_data = {key: [] for key in keys}
        for data in data_list:
            for key in keys:
                collate_data[key].append(data[key])
        # special handling
        if "spec_ints" in keys and "spec_mzs" in keys:
            # create batch_idxs
            batch_idxs, mzs, ints = batch_func(collate_data["spec_mzs"],collate_data["spec_ints"])
            collate_data["spec_mzs"] = mzs
            collate_data["spec_ints"] = ints
            collate_data["spec_batch_idxs"] = batch_idxs
            # remove from list
            keys.remove(keys.index("spec_ints"))
            keys.remove(keys.index("spec_mzs"))
        if "mol_pyg" in keys:
            # batch
            collate_data["mol_pyg"] = Batch.from_data_list(collate_data["mol_pyg"])
            # remove from list
            keys.remove(keys.index("mol_pyg"))
        # regular handling
        for key in keys:
            values = collate_data[key]
            if isinstance(values[0],list):
                # flatten
                values = flatten_lol(values)
                collate_data[key] = values
            elif isinstance(values[0],th.Tensor):
                # cat
                values = th.cat(values,dim=0)
                collate_data[key] = values
            else:
                raise TypeError(f"Unsupported type: {type(values[0])}")
        return collate_data


def SpecMolFragDataset(SpecMolDataset):

    def __init__(self,mol_df,spec_df,frag_df,split_df,,mol_params,spec_params,frag_params):

        super().__init__(mol_df,spec_df,split_df,mol_params,spec_params)
        self.frag_df = frag_df
        self.frag_params = frag_params

        # select fragments
        self.frag_df = self.frag_df[self.frag_df["spec_id"].isin(self.split_df["spec_id"])]

        # reset index
        self.frag_df = self.frag_df.reset_index(drop=True)

        # use mol_id as index for speedy access
        self.frag_df = self.frag_df.set_index("mol_id",drop=False).sort_index().rename_axis(None)

    def __getitem__(self,idx):

        spec_entry = self.spec_df.iloc[idx]
        mol_id = spec_entry["mol_id"]
        mol_entry = self.mol_df.loc[mol_id]
        spec_data = self._process_spec(spec_entry)
        mol_data = self._process_mol(mol_entry)
        frag_entry = self.frag_df.loc[mol_id]
        frag_data = self._process_frag(self,frag_entry)
        data = {**spec_data,**mol_data,**frag_data}
        return data

    def _process_frag(self,frag_entry):

        frag_data = {}
        if self.frag_params["pyg"]:
            frag_pyg = frag_entry["dag"]
            frag_pyg = get_frag_graph(frag_pyg,self.frag_params["undirected"])
            frag_data["frag_pyg"] = frag
        if self.frag_params["formula_mzs"]:
            formula_mzs = frag_entry["formula_mzs"]
            formula_mzs = formula_mzs[:,:self.frag_params["num_isotopes"]]
            frag_data["frag_formula_mzs"] = formula_mzs
        if self.frag_params["formula_probs"]:    
            formula_probs = frag_entry["formula_probs"]
            formula_probs = formula_probs[:,:self.frag_params["num_isotopes"]]
            frag_data["frag_formula_probs"] = formula_probs
        return frag_data

    @staticmethod
    def get_collate_fn():
        return SpecMolFragDataset.collate_fn

    @staticmethod
    def collate_fn(data_list):

        batch_size = len(data_list)
        keys = list(data_list[0].keys())
        collate_data = {key: [] for key in keys}
        for data in data_list:
            for key in keys:
                collate_data[key].append(data[key])
        # special handling
        if "spec_ints" in keys and "spec_mzs" in keys:
            # create batch_idxs
            batch_idxs, mzs, ints = batch_func(collate_data["spec_mzs"],collate_data["spec_ints"])
            collate_data["spec_mzs"] = mzs
            collate_data["spec_ints"] = ints
            collate_data["spec_batch_idxs"] = batch_idxs
            # remove from list
            keys.remove(keys.index("spec_ints"))
            keys.remove(keys.index("spec_mzs"))
        if "frag_dag_pyg" in keys:
            assert "mol_pyg" in keys
            assert "frag_formula_mzs" in keys
            assert "frag_formula_probs" in keys
            # process
            batch_mol_frag_data = batch_mols_frags(collate_data["mol_pyg"],collate_data["frag_dag_pyg"])
            for k,v in batch_mol_dag_data.items():
                collate_data[k] = v
            # remove from list
            keys.remove(keys.index("frag_dag_pyg"))
            keys.remove(keys.index("mol_pyg"))
            keys.remove(keys.index("frag_formula_mzs"))
            keys.remove(keys.index("frag_formula_probs"))
        # this only runs if "frag_dag_pyg" is not in keys
        if "mol_pyg" in keys:
            # batch
            collate_data["mol_pyg"] = Batch.from_data_list(collate_data["mol_pyg"])
            # remove from list
            keys.remove(keys.index("mol_pyg"))
        # regular handling
        for key in keys:
            values = collate_data[key]
            if isinstance(values[0],list):
                # flatten
                values = flatten_lol(values)
                collate_data[key] = values
            elif isinstance(values[0],th.Tensor):
                # cat
                values = th.cat(values,dim=0)
                collate_data[key] = values
            else:
                raise TypeError(f"Unsupported type: {type(values[0])}")
        return collate_data


if __name__ == "__main__":

    spec_df = pd.read_pickle("data/proc/spec_df.pkl")
    mol_df = pd.read_pickle("data/proc/mol_df.pkl")
    split_df = pd.read_pickle("data/split/debug/split_df.pkl")
    spec_params = {

    }
    mol_params = {

    }
    frag_params = {

    }