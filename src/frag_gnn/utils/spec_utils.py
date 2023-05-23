import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import  AllChem

from .formula_utils import ELECTRON_MASS

PPM = 1/1000000
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
	"""
	this will result in peaks that are really close in mass (<5ppm)
	they are probably the same peak, but our model can handle this type of ambiguity
	for now, let's keep them unmerged
	"""

	merged_peaks = {}
	# total_intensity = 0.
	for peaks in peakses:
		for mz, intensity in peaks:
			if mz in merged_peaks:
				# TODO: sum or max?
				merged_peaks[mz] += intensity
			else:
				merged_peaks[mz] = intensity
			# total_intensity += intensity
	# for mz in merged_peaks.keys():
	# 	merged_peaks[mz] /= total_intensity
	merged_peaks = sorted(list(merged_peaks.items()), key=lambda x: x[0])
	return merged_peaks


def calculate_spectrum_entropy(log_ints,batch_idxs):

	k = th.max(batch_idxs)+1
	log_norm_ints = scatter_normalize(log_ints,batch_idxs,dim=0)
	entropy = -scatter_reduce(
		src=th.exp(log_norm_ints)*log_norm_ints,
		index=batch_idxs,
		reduce="sum",
		dim=0,
		dim_size=k
	)
	return entropy


def calculate_topk_weighted_recall(true_mzs,true_logits,true_batch_idxs,pred_mzs,pred_logits,pred_batch_idxs,ppm_tolerance=10,k=10):
	"""
	topk_weighted_recall: weighted fraction of top-k true peaks that are matched by at least one top-k true peak
	topk_frac: fraction of true intensity that is explained by at least one top-k true peaks
	topk_frac is an upper bound on topk_weighted_recall
	Note: this function could be parallelized with vmap
	"""

	batch_size = th.max(true_batch_idxs)+1
	topk_weighted_recall = th.zeros([batch_size],device=true_mzs.device)
	topk_frac = th.zeros([batch_size],device=true_mzs.device)
	true_ints = th.exp(scatter_normalize(true_logits,true_batch_idxs,dim=0))
	pred_ints = th.exp(scatter_normalize(pred_logits,pred_batch_idxs,dim=0))
	for b in range(batch_size):
		b_true_mask = (true_batch_idxs==b)
		b_pred_mask = (pred_batch_idxs==b)
		if b_true_mask.sum() < k:
			b_true_argtopk = th.arange(b_true_mask.sum(),device=b_true_mask.device)
		else:
			b_true_argtopk = th.topk(true_ints[b_true_mask],k,dim=0,largest=True)[1]
		if b_pred_mask.sum() < k:
			b_pred_argtopk = th.arange(b_pred_mask.sum(),device=b_pred_mask.device)
		else:
			b_pred_argtopk = th.topk(pred_ints[b_pred_mask],k,dim=0,largest=True)[1]
		b_true_topk_mzs = true_mzs[b_true_mask][b_true_argtopk]
		b_pred_topk_mzs = pred_mzs[b_pred_mask][b_pred_argtopk]
		b_true_topk_match = calculate_match_mzs(
			b_true_topk_mzs,
			b_pred_topk_mzs,
			ppm_tolerance=ppm_tolerance,
			use_ppm=True
		)
		b_true_topk_match = th.any(b_true_topk_match,dim=1)
		b_true_ints_topk = true_ints[b_true_mask][b_true_argtopk]
		b_true_ints_topk_sum = th.sum(b_true_ints_topk)
		b_true_ints_topk_match_sum = th.sum(b_true_ints_topk[b_true_topk_match])
		b_topk_weighted_recall = b_true_ints_topk_match_sum / b_true_ints_topk_sum
		topk_weighted_recall[b] = topk_weighted_recall[b] + b_topk_weighted_recall
		b_true_ints_sum = th.sum(true_ints[b_true_mask])
		b_topk_frac = b_true_ints_topk_sum / b_true_ints_sum
		topk_frac[b] = topk_frac[b] + b_topk_frac
	return topk_weighted_recall, topk_frac


def calculate_recalls(true_mzs,true_logits,true_batch_idxs,pred_mzs,pred_logits,pred_batch_idxs,ppm_tolerance=10):
	"""
	Note: this function could be parallelized with vmap
	"""

	batch_size = th.max(true_batch_idxs)+1
	recall = th.zeros([batch_size],device=true_mzs.device)
	weighted_recall = th.zeros([batch_size],device=true_mzs.device)
	true_ints = th.exp(scatter_normalize(true_logits,true_batch_idxs,dim=0))
	pred_ints = th.exp(scatter_normalize(pred_logits,pred_batch_idxs,dim=0))
	for b in range(batch_size):
		b_true_mask = (true_batch_idxs==b)
		b_pred_mask = (pred_batch_idxs==b)
		b_true_mzs = true_mzs[b_true_mask]
		b_pred_mzs = pred_mzs[b_pred_mask]
		b_true_ints = true_ints[b_true_mask]
		b_pred_ints = pred_ints[b_pred_mask]
		b_true_match = calculate_match_mzs(
			b_true_mzs,
			b_pred_mzs,
			ppm_tolerance=ppm_tolerance,
			use_ppm=True
		)
		b_true_match = th.any(b_true_match,dim=1)
		b_recall = th.sum(b_true_match)/b_true_mzs.shape[0]
		recall[b] = recall[b] + b_recall
		b_weighted_recall = th.sum(b_true_ints[b_true_match])/th.sum(b_true_ints)
		weighted_recall[b] = weighted_recall[b] + b_weighted_recall
	return recall, weighted_recall


def calculate_match_mzs(
		true_mzs,
		pred_mzs,
		ppm_tolerance=10,
		abs_tolerance=1e-3,
		use_ppm=True,
		divisor_mz_floor=200
	):
	"""
	works with numpy arrays or torch tensors
	inspired by the function in ms-pred
	"""

	if isinstance(true_mzs,th.Tensor):
		assert isinstance(pred_mzs,th.Tensor)
		abs_func = th.abs
		copy_func = th.clone
	else:
		assert isinstance(true_mzs,np.ndarray)
		assert isinstance(pred_mzs,np.ndarray)
		abs_func = np.abs
		copy_func = np.copy
	diff_mzs = abs_func(true_mzs.reshape(-1,1)-pred_mzs.reshape(1,-1))
	if use_ppm:
		divisor_mzs = copy_func(true_mzs)
		divisor_mzs[divisor_mzs<divisor_mz_floor] = divisor_mz_floor
		diff_mzs = (diff_mzs / divisor_mzs.reshape(-1,1)) * 1e6
		min_diff_mzs = ppm_tolerance
	else:
		min_diff_mzs = abs_tolerance
	match_mzs = diff_mzs < min_diff_mzs
	return match_mzs
