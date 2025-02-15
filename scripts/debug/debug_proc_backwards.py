import pandas as pd
import argparse
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
import numpy as np


def get_adjacency_matrix(mol):
	# Compute the adjacency matrix using RDKit
	return rdmolops.GetAdjacencyMatrix(mol)

def get_atom_types(mol):
	# Get a list of atomic numbers for all atoms in the molecule
	return [atom.GetAtomicNum() for atom in mol.GetAtoms()]

def get_bond_types(mol):
	# Create a matrix to store bond types
	num_atoms = mol.GetNumAtoms()
	bond_type_matrix = np.zeros((num_atoms, num_atoms))

	for bond in mol.GetBonds():
		# Get the indices of the two atoms forming the bond
		i = bond.GetBeginAtomIdx()
		j = bond.GetEndAtomIdx()
		
		# Set the bond type (1: single, 2: double, 3: triple, 4: aromatic)
		bond_type_matrix[i, j] = bond.GetBondTypeAsDouble()
		bond_type_matrix[j, i] = bond.GetBondTypeAsDouble()  # Bond is bidirectional

	return bond_type_matrix

def compare_molecules(mol1, mol2):
	# Compare adjacency matrices
	adj_matrix1 = get_adjacency_matrix(mol1)
	adj_matrix2 = get_adjacency_matrix(mol2)
	if not np.array_equal(adj_matrix1, adj_matrix2):
		#print("The molecules have different adjacency matrices (different connectivity).")
		return False
	
	# Compare atom types
	atom_types1 = get_atom_types(mol1)
	atom_types2 = get_atom_types(mol2)
	if atom_types1 != atom_types2:
		#print("The molecules have different atom types.")
		return False

	# Compare bond types
	bond_types1 = get_bond_types(mol1)
	bond_types2 = get_bond_types(mol2)
	if not np.array_equal(bond_types1, bond_types2):
		#print("The molecules have different bond types.")
		return False

	#print("The molecules have the same atom types, bond types, and adjacency matrices.")
	return True

def get_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--new_proc_dp",type=str,default="data/proc/nist20mona23v3")
	parser.add_argument("--old_proc_dp",type=str,default="data/proc/nist20mona23v2")


	args = parser.parse_args()
	return args

if __name__ == "__main__":

	args = get_args()
 	# compare mold_df
	print(f"> compare {args.new_proc_dp} {args.old_proc_dp}")
	new_mol_df = pd.read_pickle(os.path.join(args.new_proc_dp, 'mol_df.pkl'))
	old_mol_df = pd.read_pickle(os.path.join(args.old_proc_dp, 'mol_df.pkl'))

	print("> compare shapes")
	print(f">> {args.new_proc_dp} shape:", new_mol_df.shape, f"{args.old_proc_dp} shape:", old_mol_df.shape)
	print(f">> Shape is the same: {new_mol_df.shape == old_mol_df.shape}")
	
	print("> compare mol objects by adjmatrix, bond type and atom type")
	joint_df = old_mol_df.join(new_mol_df, on='mol_id', lsuffix='_old', rsuffix='_new')
	joint_df.loc[:,'is_mol_same'] = joint_df.apply(lambda x: compare_molecules(x['mol_new'],x[('mol_old')]), axis =1)
	print(">> Reuse Frags (if any):", joint_df['is_mol_same'].all())

	# compare spec_df
	#new_spec_df = pd.read_pickle(os.path.join(args.new_proc_dp, 'spec_df.pkl'))
	#old_spec_df = pd.read_pickle(os.path.join(args.old_proc_dp, 'spec_df.pkl'))
	#print(old_mol_df.compare(old_spec_df))