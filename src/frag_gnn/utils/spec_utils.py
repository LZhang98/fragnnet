import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import  AllChem

from .formula_utils import ELECTRON_MASS

mass_to_positive_ion_mass = {
	"M+"             : lambda mass:  mass - ELECTRON_MASS,
	"M+3H"           : lambda mass:  mass/3 + 1.007276 ,
	"M+2H+Na"        : lambda mass:  mass/3 + 8.334590 ,
	"M+H+2Na"        : lambda mass:  mass/3 + 15.7661904,
	"M+3Na"          : lambda mass:  mass/3 + 22.989218,
	"M+2H"           : lambda mass:  mass/2 + 1.007276 ,
	"M+H+NH4"        : lambda mass:  mass/2 + 9.520550 ,
	"M+H+Na"         : lambda mass:  mass/2 + 11.998247,
	"M+H+K"          : lambda mass:  mass/2 + 19.985217,
	"M+ACN+2H"       : lambda mass:  mass/2 + 21.520550,
	"M+2Na"          : lambda mass:  mass/2 + 22.989218,
	"M+2ACN+2H"      : lambda mass:  mass/2 + 42.033823,
	"M+3ACN+2H"      : lambda mass:  mass/2 + 62.547097,
	"M+H"            : lambda mass:  mass + 1.007276   ,
	"M+NH4"          : lambda mass:  mass + 18.033823  ,
	"M+Na"           : lambda mass:  mass + 22.989218  ,
	"M+CH3OH+H"      : lambda mass:  mass + 33.033489  ,
	"M+K"            : lambda mass:  mass + 38.963158  ,
	"M+ACN+H"        : lambda mass:  mass + 42.033823  ,
	"M+IsoProp+H"    : lambda mass:  mass + 61.06534   ,
	"M+ACN+Na"       : lambda mass:  mass + 64.015765  ,
	"M+2K+H"         : lambda mass:  mass + 76.919040  ,
	"M+DMSO+H"       : lambda mass:  mass + 79.02122   ,
	"M+2ACN+H"       : lambda mass:  mass + 83.060370  ,
	"M+2Na-H"        : lambda mass:  mass + 44.971160  ,
	"M+IsoProp+Na+H" : lambda mass:  mass + 84.05511   ,
	"2M+H"           : lambda mass:  2*mass + 1.007276 ,
	"2M+NH4"         : lambda mass:  2*mass + 18.033823,
	"2M+Na"          : lambda mass:  2*mass + 22.989218,
	"2M+3H2O+2H"     : lambda mass:  2*mass + 28.02312 ,
	"2M+K"           : lambda mass:  2*mass + 38.963158,
	"2M+ACN+H"       : lambda mass:  2*mass + 42.033823,
	"2M+ACN+Na"      : lambda mass:  2*mass + 64.015765,
	"M+Li"           : lambda mass:  mass + 6.9400376247955,
    }
    
mass_to_negative_ion_mass = {
	"M-"            : lambda mass:  mass + ELECTRON_MASS,
	"M-3H"          : lambda mass:  mass/3 - 1.007276  ,
	"M-2H"          : lambda mass:  mass/2 - 1.007276  ,
	"M-H20-H"       : lambda mass:  mass - 19.01839    ,
	"M-H"           : lambda mass:  mass - 1.007276    ,
	"M+Na-2H"       : lambda mass:  mass + 20.974666   ,
	"M+Cl"          : lambda mass:  mass + 34.969402   ,
	"M+K-2H"        : lambda mass:  mass + 36.948606   ,
	"M+FA-H"        : lambda mass:  mass + 44.998201   ,
	"M+Hac-H"       : lambda mass:  mass + 59.013851   ,
	"M+Br"          : lambda mass:  mass + 78.918885   ,
	"M+TFA-H"       : lambda mass:  mass + 112.985586  ,
	"2M-H"          : lambda mass:  2*mass - 1.007276  ,
	"2M+FA-H"       : lambda mass:  2*mass + 44.998201 ,
	"2M+Hac-H"      : lambda mass:  2*mass + 59.013851 ,
	"3M-H"          : lambda mass:  3*mass - 1.007276  ,
	"M+HCOO"        : lambda mass:  mass + 44.998201  ,
	"M+CH3COO"      : lambda mass:  mass + 59.0133}

positive_ion_mass_to_mass = {
	"M+"             : lambda mass:  mass + ELECTRON_MASS,
	"M+3H"           : lambda mass:  mass/3 - 1.007276 ,
	"M+2H+Na"        : lambda mass:  mass/3 - 8.334590 ,
	"M+H+2Na"        : lambda mass:  mass/3 - 15.7661904,
	"M+3Na"          : lambda mass:  mass/3 - 22.989218,
	"M+2H"           : lambda mass:  mass/2 - 1.007276 ,
	"M+H+NH4"        : lambda mass:  mass/2 - 9.520550 ,
	"M+H+Na"         : lambda mass:  mass/2 - 11.998247,
	"M+H+K"          : lambda mass:  mass/2 - 19.985217,
	"M+ACN+2H"       : lambda mass:  mass/2 - 21.520550,
	"M+2Na"          : lambda mass:  mass/2 - 22.989218,
	"M+2ACN+2H"      : lambda mass:  mass/2 - 42.033823,
	"M+3ACN+2H"      : lambda mass:  mass/2 - 62.547097,
	"M+H"            : lambda mass:  mass - 1.007276   ,
	"M+NH4"          : lambda mass:  mass - 18.033823  ,
	"M+Na"           : lambda mass:  mass - 22.989218  ,
	"M+CH3OH+H"      : lambda mass:  mass - 33.033489  ,
	"M+K"            : lambda mass:  mass - 38.963158  ,
	"M+ACN+H"        : lambda mass:  mass - 42.033823  ,
	"M+IsoProp+H"    : lambda mass:  mass - 61.06534   ,
	"M+ACN+Na"       : lambda mass:  mass - 64.015765  ,
	"M+2K+H"         : lambda mass:  mass - 76.919040  ,
	"M+DMSO+H"       : lambda mass:  mass - 79.02122   ,
	"M+2ACN+H"       : lambda mass:  mass - 83.060370  ,
	"M+2Na-H"        : lambda mass:  mass - 44.971160  ,
	"M+IsoProp+Na+H" : lambda mass:  mass - 84.05511   ,
	"2M+H"           : lambda mass:  2*mass - 1.007276 ,
	"2M+NH4"         : lambda mass:  2*mass - 18.033823,
	"2M+Na"          : lambda mass:  2*mass - 22.989218,
	"2M+3H2O+2H"     : lambda mass:  2*mass - 28.02312 ,
	"2M+K"           : lambda mass:  2*mass - 38.963158,
	"2M+ACN+H"       : lambda mass:  2*mass - 42.033823,
	"2M+ACN+Na"      : lambda mass:  2*mass - 64.015765,
	"M+Li"           : lambda mass:  mass - 6.9400376247955,
}
    
negative_ion_mass_to_mass = {
    "M-"            : lambda mass:  mass - ELECTRON_MASS,
    "M-3H"          : lambda mass:  mass/3 + 1.007276  ,
    "M-2H"          : lambda mass:  mass/2 + 1.007276  ,
    "M-H20-H"       : lambda mass:  mass + 19.01839    ,
    "M-H"           : lambda mass:  mass + 1.007276    ,
    "M+Na-2H"       : lambda mass:  mass - 20.974666   ,
    "M+Cl"          : lambda mass:  mass - 34.969402   ,
    "M+K-2H"        : lambda mass:  mass - 36.948606   ,
    "M+FA-H"        : lambda mass:  mass - 44.998201   ,
    "M+Hac-H"       : lambda mass:  mass - 59.013851   ,
    "M+Br"          : lambda mass:  mass - 78.918885   ,
    "M+TFA-H"       : lambda mass:  mass - 112.985586  ,
    "2M-H"          : lambda mass:  2*mass + 1.007276  ,
    "2M+FA-H"       : lambda mass:  2*mass - 44.998201 ,
    "2M+Hac-H"      : lambda mass:  2*mass - 59.013851 ,
    "3M-H"          : lambda mass:  3*mass + 1.007276  ,
    "M+HCOO"        : lambda mass:  mass - 44.998201  ,
    "M+CH3COO"      : lambda mass:  mass - 59.0133}


def neutral_mass_to_ion_mass(neutral_mass, adduct_type):
	"""[summary]

	Args:
		neutral_mass ([type]): [description]
		adduct_type ([type]): [description]

	Returns:
		[type]: [description]
	"""
	ion_name = adduct_type.split(']')[0][1:]
	if adduct_type.endswith('+') or adduct_type == '[M+]':
		return mass_to_positive_ion_mass[ion_name](neutral_mass)
	elif adduct_type.endswith('-') or adduct_type == '[M-]':
		return mass_to_negative_ion_mass[ion_name](neutral_mass)
	else:
		print('Can not compute ion mass for adduct_type', adduct_type)


def ion_mass_to_neutral_mass(neutral_mass, adduct_type):
	"""[summary]

	Args:
		neutral_mass ([type]): [description]
		adduct_type ([type]): [description]

	Returns:
		[type]: [description]
	"""
	ion_name = adduct_type.split(']')[0][1:]
	if adduct_type.endswith('+') or adduct_type == '[M+]':
		return positive_ion_mass_to_mass[ion_name](neutral_mass)
	elif adduct_type.endswith('-') or adduct_type == '[M-]':
		return negative_ion_mass_to_mass[ion_name](neutral_mass)
	else:
		print('Can not compute ion mass for adduct_type', adduct_type)

def create_mol_by_inchi_or_smiles(smiles_or_inchi: str):
	"""[summary]

	Args:
		chem_id (str): [description]

	Returns:
		[type]: [description]
	"""
	# get inchi and inchi keys
	mol = None
	if smiles_or_inchi.startswith("InChI"):
		# if we have a inchi or standard inchi
		mol = Chem.inchi.MolFromInchi(smiles_or_inchi)
	else:
		# if we have a smiles
		mol = Chem.MolFromSmiles(smiles_or_inchi)

	if mol == None:
		raise ValueError('Can not create rdkit mol', smiles_or_inchi)
	return mol


def get_inchikey_by_chemid(smiles_or_inchi: str):
	mol = create_mol_by_inchi_or_smiles(smiles_or_inchi)
	return Chem.inchi.MolToInchiKey(mol)


def get_mol_mw_by_chemid(smiles_or_inchi: str):
	mol = create_mol_by_inchi_or_smiles(smiles_or_inchi)
	return Descriptors.ExactMolWt(mol)


def get_formula_by_chemid(smiles_or_inchi: str):
	mol = create_mol_by_inchi_or_smiles(smiles_or_inchi)
	return Chem.rdMolDescriptors.CalcMolFormula(mol)

def get_mass_tol(mass:float, ppm_tol:float = None, abs_tol:float = None) -> float:
	"""_summary_

	Args:
		mass (float): _description_
		ppm_tol (float, optional): _description_. Defaults to None.
		abs_tol (float, optional): _description_. Defaults to None.

	Returns:
		float: _description_
	"""
	assert ppm_tol is None and abs_tol is None
	if abs_tol is not None:
		return abs_tol
	else:
		return mass * ppm_tol * PPM

def get_mass_range(mass:float, ppm_tol:float = None, abs_tol:float = None) -> tuple(float,float):
	"""_summary_

	Args:
		float (_type_): _description_
		mass (_type_, optional): _description_. Defaults to None, abs_tol:float = None)->tuple(float.

	Returns:
		_type_: _description_
	"""
	mass_tol = get_mass_tol(mass, ppm_tol, abs_tol)
	return min(mass - mass_tol,0), mass + mass_tol 


def bin_func(mzs,ints,mz_max,mz_bin_res,ints_thresh,return_index):

	# import pdb; pdb.set_trace()
	mzs = np.array(mzs,dtype=np.float32)
	bins = np.arange(mz_bin_res,mz_max+mz_bin_res,step=mz_bin_res).astype(np.float32)
	bin_idx = np.searchsorted(bins,mzs,side="right")
	if return_index:
		return bin_idx.tolist()
	else:
		ints = np.array(ints,dtype=np.float32)
		bin_spec = np.zeros([len(bins)],dtype=np.float32)
		for i in range(len(mzs)):
			if bin_idx[i] < len(bin_spec) and ints[i] >= ints_thresh:
				bin_spec[bin_idx[i]] = max(bin_spec[bin_idx[i]],ints[i])
		if np.all(bin_spec == 0.):
			print("> warning: bin_spec is all zeros!")
			bin_spec[-1] = 1.
		return bin_spec

def merge_sparse_specs(*peakses):

	merged_peaks = {}
	for peaks in peakses:
		for mz, int in peaks:
			if mz in merged_peaks:
				merged_peaks[mz] += int
			else:
				merged_peaks = int
	merged_peaks = sorted(list(merged_peaks.items()), key=lambda x: x[0])
	return merged_peaks
