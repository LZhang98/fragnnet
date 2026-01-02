"""
Example usage of hard negative sampler with various loss functions.

Includes:
- Triplet loss with explicit hard negatives
- Modified CLIP/contrastive loss with hard negative weighting
- Training loop example
- Batch data organization examples
"""

import torch as th
import torch.nn.functional as F
from typing import Dict, Optional


def triplet_loss_with_hard_negatives(
	anchor_embeddings: th.Tensor,
	hard_neg_embeddings: th.Tensor,  # [batch_size, num_hard_negs, embedding_dim]
	random_neg_embeddings: th.Tensor,  # [batch_size, num_random_negs, embedding_dim]
	temperature: float = 0.07,
	margin: float = 0.1,
) -> th.Tensor:
	"""
	Triplet loss with explicit hard negatives.
	
	For each anchor, computes:
		loss = max(0, hard_neg_sim - random_neg_sim + margin)
	
	This encourages hard negatives to be pushed away more than random negatives.
	
	Args:
		anchor_embeddings: [batch_size, embedding_dim]
		hard_neg_embeddings: [batch_size, num_hard_negs, embedding_dim]
		random_neg_embeddings: [batch_size, num_random_negs, embedding_dim]
		temperature: Temperature scaling (unused here, for API consistency)
		margin: Triplet margin
	
	Returns:
		Scalar loss
	"""
	batch_size = anchor_embeddings.shape[0]
	
	# Normalize embeddings for cosine similarity
	anchor_norm = F.normalize(anchor_embeddings, p=2, dim=1)  # [B, D]
	hard_neg_norm = F.normalize(hard_neg_embeddings, p=2, dim=2)  # [B, K_hard, D]
	random_neg_norm = F.normalize(random_neg_embeddings, p=2, dim=2)  # [B, K_rand, D]
	
	# Compute cosine similarities (dot product of normalized vectors)
	# hard_neg_sims: [B, K_hard]
	hard_neg_sims = th.bmm(anchor_norm.unsqueeze(1), hard_neg_norm.transpose(1, 2)).squeeze(1)
	# Take minimum (hardest negative)
	hard_neg_min = th.min(hard_neg_sims, dim=1)[0]  # [B]
	
	# random_neg_sims: [B, K_rand]
	random_neg_sims = th.bmm(anchor_norm.unsqueeze(1), random_neg_norm.transpose(1, 2)).squeeze(1)
	# Take maximum (easiest random negative)
	random_neg_max = th.max(random_neg_sims, dim=1)[0] if random_neg_sims.shape[1] > 0 else th.zeros(batch_size, device=anchor_embeddings.device)
	
	# Triplet loss
	loss = F.relu(hard_neg_min - random_neg_max + margin)
	
	return loss.mean()


def contrastive_loss_with_hard_negatives(
	anchor_embeddings: th.Tensor,
	hard_neg_embeddings: th.Tensor,
	random_neg_embeddings: th.Tensor,
	temperature: float = 0.07,
	hard_neg_weight: float = 2.0,
) -> th.Tensor:
	"""
	Modified CLIP/InfoNCE loss with explicit hard negative weighting.
	
	For each anchor, compute log-softmax over all negatives,
	with hard negatives contributing more to the loss (higher weight).
	
	Args:
		anchor_embeddings: [batch_size, embedding_dim]
		hard_neg_embeddings: [batch_size, num_hard_negs, embedding_dim]
		random_neg_embeddings: [batch_size, num_random_negs, embedding_dim]
		temperature: Temperature for scaling logits
		hard_neg_weight: Weight for hard negatives (>1 emphasizes hard negs)
	
	Returns:
		Scalar loss
	"""
	batch_size = anchor_embeddings.shape[0]
	device = anchor_embeddings.device
	
	# Normalize
	anchor_norm = F.normalize(anchor_embeddings, p=2, dim=1)  # [B, D]
	hard_neg_norm = F.normalize(hard_neg_embeddings, p=2, dim=2)  # [B, K_hard, D]
	random_neg_norm = F.normalize(random_neg_embeddings, p=2, dim=2)  # [B, K_rand, D]
	
	# Compute logits: anchor @ neg^T / temperature
	hard_neg_logits = th.bmm(
		anchor_norm.unsqueeze(1), hard_neg_norm.transpose(1, 2)
	).squeeze(1) / temperature  # [B, K_hard]
	
	random_neg_logits = th.bmm(
		anchor_norm.unsqueeze(1), random_neg_norm.transpose(1, 2)
	).squeeze(1) / temperature if random_neg_embeddings.shape[1] > 0 else th.empty((batch_size, 0), device=device)
	
	# Stack logits and apply hard negative weighting
	if hard_neg_logits.shape[1] > 0 and random_neg_logits.shape[1] > 0:
		all_logits = th.cat([hard_neg_logits, random_neg_logits], dim=1)
		# Weight hard negatives by duplicating/reweighting
		num_hard = hard_neg_logits.shape[1]
		weights = th.cat([
			hard_neg_weight * th.ones(batch_size, num_hard, device=device),
			th.ones(batch_size, random_neg_logits.shape[1], device=device)
		], dim=1)
	elif hard_neg_logits.shape[1] > 0:
		all_logits = hard_neg_logits
		weights = th.ones_like(all_logits)
	else:
		return th.tensor(0.0, device=device)
	
	# Weighted softmax
	weighted_logits = all_logits * weights
	log_probs = F.log_softmax(weighted_logits, dim=1)
	
	# Loss: negative log-likelihood
	# We want hard negatives to have HIGH loss (low probability)
	num_hard = hard_neg_logits.shape[1]
	if num_hard > 0:
		hard_neg_loss = -log_probs[:, :num_hard].sum(dim=1).mean()
	else:
		hard_neg_loss = th.tensor(0.0, device=device)
	
	return hard_neg_loss


def nce_loss_with_hard_negatives(
	anchor_embeddings: th.Tensor,
	hard_neg_embeddings: th.Tensor,
	random_neg_embeddings: th.Tensor,
	temperature: float = 0.07,
	hard_neg_scale: float = 1.0,
	random_neg_scale: float = 1.0,
) -> th.Tensor:
	"""
	Noise Contrastive Estimation (NCE) loss with explicit hard negative handling.
	
	Args:
		anchor_embeddings: [batch_size, embedding_dim]
		hard_neg_embeddings: [batch_size, num_hard_negs, embedding_dim]
		random_neg_embeddings: [batch_size, num_random_negs, embedding_dim]
		temperature: Temperature for scaling
		hard_neg_scale: Scale factor for hard negative logits
		random_neg_scale: Scale factor for random negative logits
	
	Returns:
		Scalar NCE loss
	"""
	batch_size = anchor_embeddings.shape[0]
	device = anchor_embeddings.device
	
	# Normalize
	anchor_norm = F.normalize(anchor_embeddings, p=2, dim=1)
	hard_neg_norm = F.normalize(hard_neg_embeddings, p=2, dim=2)
	random_neg_norm = F.normalize(random_neg_embeddings, p=2, dim=2)
	
	# Compute similarities
	hard_neg_sims = th.bmm(anchor_norm.unsqueeze(1), hard_neg_norm.transpose(1, 2)).squeeze(1) / temperature
	random_neg_sims = th.bmm(anchor_norm.unsqueeze(1), random_neg_norm.transpose(1, 2)).squeeze(1) / temperature if random_neg_embeddings.shape[1] > 0 else th.empty((batch_size, 0), device=device)
	
	# NCE: log(1 + sum(exp(hard_negs) + exp(random_negs)))
	all_sims = th.cat([
		hard_neg_sims * hard_neg_scale,
		random_neg_sims * random_neg_scale if random_neg_sims.shape[1] > 0 else th.empty((batch_size, 0), device=device)
	], dim=1)
	
	nce_loss = th.log(1.0 + th.logsumexp(all_sims, dim=1)).mean()
	
	return nce_loss


def organize_batch_for_pairwise_loss(batch: Dict, num_hard_negatives: int, num_random_negatives: int) -> Dict:
	"""
	Reorganize hard negative sampler batch into pairwise structure.
	
	Useful for pairwise cross-entropy loss where you want to compute
	loss over all pairs within a batch.
	
	Args:
		batch: Batch from FormulaHardNegativeCollator
		num_hard_negatives: Number of hard negatives in batch
		num_random_negatives: Number of random negatives in batch
	
	Returns:
		Dict with keys:
		- anchors: [batch_size, ...]
		- negatives: [batch_size * (K_hard + K_rand), ...]
		- neg_types: [batch_size * (K_hard + K_rand)] (0=hard, 1=random)
		- batch_indices: [batch_size * (K_hard + K_rand)] (anchor idx for each negative)
	"""
	anchors_list = []
	negatives_list = []
	neg_types_list = []
	batch_indices_list = []
	
	# Collect anchors
	for key in batch:
		if key.startswith("anchor_") and not key.startswith("anchor_hard") and not key.startswith("anchor_random"):
			anchors_list.append(key)
	
	# For each hard negative, append (anchor, neg) pair
	for i in range(num_hard_negatives):
		neg_key_suffix = f"hard_neg_{i}"
		for key in batch:
			if key.startswith(neg_key_suffix + "_"):
				neg_types_list.append(0)  # hard negative
				batch_indices_list.append(i)
	
	# For each random negative, append (anchor, neg) pair
	for i in range(num_random_negatives):
		neg_key_suffix = f"random_neg_{i}"
		for key in batch:
			if key.startswith(neg_key_suffix + "_"):
				neg_types_list.append(1)  # random negative
				batch_indices_list.append(i + num_hard_negatives)
	
	return {
		"neg_types": th.tensor(neg_types_list),
		"batch_indices": th.tensor(batch_indices_list),
	}


def example_training_loop(
	train_loader,
	model,
	optimizer,
	num_epochs: int = 10,
	loss_fn: str = "contrastive",
	device: str = "cuda" if th.cuda.is_available() else "cpu",
):
	"""
	Example training loop using hard negative sampler and loss functions.
	
	Args:
		train_loader: DataLoader from create_formula_hard_negative_loader
		model: Your spectrum-molecule embedding model
		optimizer: PyTorch optimizer
		num_epochs: Number of epochs
		loss_fn: One of ["triplet", "contrastive", "nce"]
		device: Device to train on
	"""
	model.to(device)
	model.train()
	
	loss_functions = {
		"triplet": triplet_loss_with_hard_negatives,
		"contrastive": contrastive_loss_with_hard_negatives,
		"nce": nce_loss_with_hard_negatives,
	}
	
	if loss_fn not in loss_functions:
		raise ValueError(f"loss_fn must be one of {list(loss_functions.keys())}")
	
	loss_fn_func = loss_functions[loss_fn]
	
	for epoch in range(num_epochs):
		total_loss = 0
		num_batches = 0
		
		for batch_idx, batch in enumerate(train_loader):
			# Batch structure from FormulaHardNegativeCollator:
			# - anchor_{key}: Anchor spectrum/molecule data
			# - hard_neg_{i}_{key}: Hard negative data
			# - random_neg_{i}_{key}: Random negative data
			# - roles: Role of each item
			# - num_hard_negatives, num_random_negatives: Counts
			
			num_hard = batch["num_hard_negatives"]
			num_random = batch["num_random_negatives"]
			
			# Example: assuming model.encode() takes relevant batch keys
			# You'll need to adapt this based on your model interface
			
			# Get anchor embeddings
			anchor_keys = {k.replace("anchor_", ""): v for k, v in batch.items() if k.startswith("anchor_")}
			anchor_emb = model.encode_spectrum(anchor_keys)  # [B, D]
			
			# Get hard negative embeddings [B, K_hard, D]
			hard_neg_embs = []
			for i in range(num_hard):
				hard_neg_keys = {
					k.replace(f"hard_neg_{i}_", ""): v 
					for k, v in batch.items() 
					if k.startswith(f"hard_neg_{i}_")
				}
				hard_neg_embs.append(model.encode_spectrum(hard_neg_keys))
			hard_neg_embs = th.stack(hard_neg_embs, dim=1) if hard_neg_embs else th.empty((anchor_emb.shape[0], 0, anchor_emb.shape[1]), device=device)
			
			# Get random negative embeddings [B, K_rand, D]
			random_neg_embs = []
			for i in range(num_random):
				random_neg_keys = {
					k.replace(f"random_neg_{i}_", ""): v 
					for k, v in batch.items() 
					if k.startswith(f"random_neg_{i}_")
				}
				random_neg_embs.append(model.encode_spectrum(random_neg_keys))
			random_neg_embs = th.stack(random_neg_embs, dim=1) if random_neg_embs else th.empty((anchor_emb.shape[0], 0, anchor_emb.shape[1]), device=device)
			
			# Compute loss
			loss = loss_fn_func(anchor_emb, hard_neg_embs, random_neg_embs)
			
			# Backward
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			total_loss += loss.item()
			num_batches += 1
			
			if (batch_idx + 1) % 10 == 0:
				print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}")
		
		avg_loss = total_loss / num_batches
		print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")


# ============================================================================
# Batch structure reference
# ============================================================================

BATCH_STRUCTURE_EXAMPLE = """
Example batch structure returned by FormulaHardNegativeCollator:

{
    'batch_indices': [0, 15, 32, 78, 45],  # Original dataset indices
    'roles': [0, 1, 1, 2],  # 0=anchor, 1=hard_neg, 2=random_neg
    'num_hard_negatives': 2,
    'num_random_negatives': 1,
    
    # Anchor keys (no index suffix)
    'anchor_spec_mzs': tensor([...]),
    'anchor_spec_intensities': tensor([...]),
    'anchor_mol_graph': Data(...),
    'anchor_mol_id': [...],
    
    # Hard negative keys (indexed by position in hard_neg triplet)
    'hard_neg_0_spec_mzs': tensor([...]),
    'hard_neg_0_spec_intensities': tensor([...]),
    'hard_neg_0_mol_graph': Data(...),
    
    'hard_neg_1_spec_mzs': tensor([...]),
    'hard_neg_1_spec_intensities': tensor([...]),
    'hard_neg_1_mol_graph': Data(...),
    
    # Random negative keys
    'random_neg_0_spec_mzs': tensor([...]),
    'random_neg_0_spec_intensities': tensor([...]),
    'random_neg_0_mol_graph': Data(...),
}
"""
