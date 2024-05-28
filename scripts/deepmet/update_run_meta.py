import argparse
import os
import json
import shutil
import yaml
import wandb

def get_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--wandb_dir",type=str,default="./wandb/")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = get_args()

	wandb_dir = args.wandb_dir

 
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
			continue
		else:
			with open(config_fp) as f:
				config = yaml.load(f, Loader=yaml.FullLoader)
			wandb_name = config["wandb_name"]
			wandb_grp = config["wandb_group"] if 'wandb_group' in config else None
			wandb_tags = config["wandb_tags"] if 'wandb_tags' in config else []
			run = wandb.Api().run(f"frag-gnn/frag-gnn/{wandb_id}")
			run.group = wandb_grp
			run.name = wandb_name
			run.tags = wandb_tags
			run.update()
   			#run.update()