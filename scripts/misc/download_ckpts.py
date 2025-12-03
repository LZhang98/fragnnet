import shutil
import tempfile
import glob
import os
import wandb

def get_ckpt_from_wandb(saved_dp, run_id, 
						entity='frag-gnn', 
						project='frag-gnn',
						use_cached = False,
      					best_only=True):
	# return ckpt
	cached_ckpts = glob.glob(os.path.join(saved_dp, f'*_{run_id}.ckpt'))
	if use_cached and len(cached_ckpts) == 1:
		print(f"> found cached ckpt {cached_ckpts[0]} for {run_id}")
		return cached_ckpts[0]

	api = wandb.Api()
	run = api.run(f"{entity}/{project}/{run_id}")
	#print(run.id, run.name, run.tags)
	print(f"> Processing model files for run {run.id} {run.name}")
	model_tag = f"{run.name}_{run_id}"
	#model_save_dp = f"{saved_dp}/{model_tag}"
	os.makedirs(saved_dp, exist_ok=True)
	best_ckpt_file, best_ckpt_epoch_num = None, 0
	ckpt_files = []
	saved_filenames = []
	for file in run.files():
		if file.name.startswith("ckpt/model-epoch="):
			ckpt_files.append(file.name)
			epoch_num = int(file.name.removeprefix("ckpt/model-epoch=").removesuffix(".ckpt"))
			saved_filenames.append(f"{model_tag}_epoch={epoch_num}.ckpt")
			#print(epoch_num)
			if best_ckpt_file is None or epoch_num > best_ckpt_epoch_num:
				best_ckpt_file = file.name
				best_ckpt_epoch_num = epoch_num
    
	if best_only:
		ckpt_files = [best_ckpt_file]
		saved_filenames = [f"{model_tag}.ckpt"]
  
	if len(ckpt_files) == 0:
		print("> Skip, no ckpt files found")
		return

	for ckpt_file, saved_filename in zip(ckpt_files, saved_filenames):
		ckpt_fp = f"{saved_dp}/{saved_filename}"
		if os.path.isfile(ckpt_fp) and use_cached:
			print(f"> found cached ckpt {ckpt_fp}, skip downloading")
			continue

		with tempfile.TemporaryDirectory() as tmp_dir:
			#if not os.path.exists(ckpt_fp):
			print(f"> Downloading ckpt {ckpt_file}")
			run.file(ckpt_file).download(root=tmp_dir,replace=False)
			print(f"> Save ckpt {ckpt_fp}")
			shutil.copy(f'{tmp_dir}/{ckpt_file}', ckpt_fp)

if __name__ == "__main__":

	save_dp = "./saved_ckpts/"
	os.makedirs(save_dp, exist_ok=True)
	# deepmet ckpt
	run_ids = [#"",
            "dq50hzdv", #cv1
            "764tpm7g",	#cv2
            "nkfzye72",	#cv3
            "77hbb6fy", #cv4
            "yp1ntozu" #cv_all
            ]
	for run_id in run_ids:
		get_ckpt_from_wandb(save_dp, run_id, best_only=True)
  
	# pretraining ckpt
	run_ids = [#"",
            "edd2j4yy", #m30
            "zpz5zl0p", #m30
            "6wrmthk2", #
            "4968bah0", #m30_sc
            ]
	for run_id in run_ids:
		get_ckpt_from_wandb(save_dp, run_id, best_only=True)