import shutil
import tempfile
import glob
import os

def get_best_ckpt_from_wandb(saved_dp, run_id, 
						entity='frag-gnn', 
						project='frag-gnn',
						use_cached = False):
	# return ckpt
	cached_ckpts = glob.glob(os.path.join(saved_dp, f'*_{run_id}.ckpt'))
	if use_cached and len(cached_ckpts) == 1:
		print(f"> found cached ckpt {cached_ckpts[0]} for {run_id}")
		return cached_ckpts[0]

	import wandb
	api = wandb.Api()
	run = api.run(f"{entity}/{project}/{run_id}")
	#print(run.id, run.name, run.tags)
	print(f"> Processing model files for run {run.id} {run.name}")
	model_tag = f"{run.name}_{run_id}"
	#model_save_dp = f"{saved_dp}/{model_tag}"
	os.makedirs(saved_dp, exist_ok=True)
	ckpt_file, ckpt_epoch_num = None, None
	for file in run.files():
		if file.name.startswith("ckpt/model-epoch="):
			epoch_num = int(file.name.removeprefix("ckpt/model-epoch=").removesuffix(".ckpt"))
			#print(epoch_num)
			if ckpt_file is None or epoch_num > ckpt_epoch_num:
				ckpt_epoch_num = epoch_num
				ckpt_file = file.name

	if ckpt_file is None or ckpt_epoch_num is None:
		print("> Skip, ckpt_file is None or ckpt_epoch_num is None")
	
	#assert not ckpt_file is None
	#assert not ckpt_epoch_num is None
	ckpt_fp = f"{saved_dp}/{model_tag}.ckpt"
	if os.path.isfile(ckpt_fp) and use_cached:
		return ckpt_fp

	with tempfile.TemporaryDirectory() as tmp_dir:
		#if not os.path.exists(ckpt_fp):
		print(f"> Downloading ckpt {ckpt_file}")
		run.file(ckpt_file).download(root=tmp_dir,replace=False)
		print(f"> Save ckpt {ckpt_fp}")
		shutil.copy(f'{tmp_dir}/{ckpt_file}', ckpt_fp)
	return ckpt_fp

if __name__ == "__main__":

	save_dp = "./saved_ckpts/"
	os.makedirs(save_dp, exist_ok=True)
	run_id = "ikohsc16"
	get_best_ckpt_from_wandb(save_dp, run_id)