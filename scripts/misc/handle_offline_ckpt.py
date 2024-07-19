import argparse
import os
import json
import shutil
import yaml

def get_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--wandb_dir",type=str,default="./wandb/")
	parser.add_argument("--save_base_dp", type=str,default="ckpts")
	parser.add_argument("--model_types", type=str,nargs="+",default=['massformer','neims'], help="allowed model types")
	parser.add_argument("--operation", type=str, default="dryrun",choices=['dryrun','copy','move'], help="True copy ckpt to save_dp")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = get_args()

	wandb_dir = args.wandb_dir
	save_base_dp = args.save_base_dp
 
	wandb_offline_dirnames = [d for d in os.listdir(wandb_dir) if d.startswith("offline")]
	for dirname in wandb_offline_dirnames:
		wandb_offline_dir = os.path.join(wandb_dir, dirname)
		wandb_id = dirname.split('-')[-1]
		ckpt_dir = os.path.join(wandb_offline_dir, "files", "ckpt")
		print(f"> proccessing {wandb_offline_dir}")
		print(f"> wandb_id {wandb_id}")
		# check wandb-metadata.json
		wandb_metadata_fp = os.path.join(wandb_offline_dir, "files", "wandb-metadata.json")
		if not os.path.isfile(wandb_metadata_fp):
			print(f"> skip {wandb_id} due to no wandb-metadata.json")
			continue
				
		with open(wandb_metadata_fp) as f:
			wandb_metadata = json.load(f)
			#print(wandb_metadata)
		
		if not wandb_metadata["program"].endswith("run_pl_model_fit.py"):
			print(f"> skip {wandb_id} due to not a training run")
			continue
      
		if not os.path.isdir(ckpt_dir):
			print(f"> skip {wandb_id} due to no a ckpt_dir")
			continue
		
		trainig_args = wandb_metadata['args']
		config_fp = None
		for k,v in zip(trainig_args[0::2],trainig_args[1::2]):
			if k == "-c":
				config_fp = v
				break

		# check config.yaml
		if config_fp is None or not os.path.isfile(config_fp):
			print(f"> skip, config_fp: {config_fp} can not be processed")
			continue
		else:
			with open(config_fp) as f:
				config = yaml.load(f, Loader=yaml.FullLoader)
			wandb_name = config["wandb_name"]
			wandb_grp = config["wandb_group"] if 'wandb_group' in config else None
			model_type = config["model_type"] if 'model_type' in config else "frag_gnn"

		#print(wandb_id, wandb_name, wandb_grp, model_type)
		#print(wandb_offline_dir)
		if model_type not in args.model_types:
			print(f"> skip {wandb_id} due to model_type {model_type} not in {args.model_types}")
			continue

		print(f"> wandb_grp {wandb_grp}")
		print(f"> wandb_name {wandb_name}")
		print(f"> model_type {model_type}")

		ckpt_files = [f for f in os.listdir(ckpt_dir)]
		best_ckpt_fp = None
		last_ckpt_fp = None
		ckpt_epoch_num = 0
		for ckpt_file in ckpt_files:
			if ckpt_file.startswith("model-epoch="):
				epoch_num = int(ckpt_file.removeprefix("model-epoch=").removesuffix(".ckpt"))
				if ckpt_file is None or epoch_num > ckpt_epoch_num:
					ckpt_epoch_num = epoch_num
					best_ckpt_fp = os.path.join(ckpt_dir, ckpt_file)
			elif ckpt_file.endswith("last.ckpt"):
				last_ckpt_fp = os.path.join(ckpt_dir, ckpt_file)
		
		if best_ckpt_fp is None and last_ckpt_fp is None:
			print(f"> skip {wandb_id} due to no ckpt files")
			continue

		save_dp = os.path.join(save_base_dp, wandb_grp, f"{wandb_name}_{wandb_id}") if wandb_grp is not None else os.path.join(save_base_dp, f"{wandb_name}_{wandb_id}")
		os.makedirs(save_dp, exist_ok=True)
		ckpt_dict = {
			"best": best_ckpt_fp,
			"last": last_ckpt_fp
		}
		for ckpt_type in ckpt_dict:
			src_ckpt_fp = ckpt_dict[ckpt_type]
			if src_ckpt_fp is None:
				continue
	
			save_fp = os.path.join(save_dp, f"{wandb_name}_{ckpt_type}.ckpt")
			if args.operation == "dryrun":
				print(f"find {src_ckpt_fp}, move/copy to {save_fp}")
			elif args.operation == "copy":
				print(f"copying {src_ckpt_fp}, to {save_fp}")
				shutil.copy(src_ckpt_fp, save_fp)
			elif  args.operation == "move":
				print(f"copying {src_ckpt_fp}, to {save_fp}")
				shutil.move(src_ckpt_fp, save_fp)

		if args.operation != "dryrun":
			operation_dict = {}
			operation_dict['best_src'] = best_ckpt_fp
			operation_dict['last_src'] = last_ckpt_fp
			operation_dict['wandb_id'] = wandb_id
			operation_dict['type'] = args.operation
			with open(os.path.join(save_dp, "operation_record.json"),"w") as f:
				json.dump(operation_dict,f,indent=4)