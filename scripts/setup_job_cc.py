import os
import argparse
import subprocess

def main(args):

	# create folders
	os.makedirs(f"{os.path.abspath(args.frag_gnn_dp)}/wandb", exist_ok=True)
	os.makedirs(f"{os.path.abspath(args.script_dp)}", exist_ok=True)
	os.makedirs(f"{os.path.abspath(args.job_dp)}", exist_ok=True)

	script_lines = []
	# account, ram, cpu, gpu, and etc
	if not args.interactive:
		script_lines.extend(
			[
				"#!/bin/bash",
				f"#SBATCH --account={args.cc_account}",			
				"#SBATCH --nodes=1",
				f"#SBATCH --job-name={args.job_name}",
				f"#SBATCH --mem={args.mem}G",
				f"#SBATCH --cpus-per-task={args.cpu}",
				f"#SBATCH --time=0-{args.time}:00:00",
				f"#SBATCH --mail-user={args.cc_email}",
				"#SBATCH --mail-type=ALL"
			]
		)
		# add gpus
		if args.gpu > 0:
			# eg. --gpus-per-node=v100:1 or --gpus-per-node=1
			script_lines.append(f"#SBATCH --gpus-per-node={args.gpu}")


		# config where to save job output and job error
		# append to existing onces
		script_lines.extend(
			[
				f"#SBATCH --output={os.path.abspath(args.job_dp)}/{args.job_name}_%j.out",
				f"#SBATCH --error={os.path.abspath(args.job_dp)}/{args.job_name}_%j.err",
				"#SBATCH --open-mode=append"
			]
		)
		# start recording 	
		script_lines.extend(
			[
				"", 
				"# record start",
				"echo `date`: Job ${SLURM_JOB_ID} is allocated resources"
			]
		)
		
	else:
		script_lines.append('#allocate job')
		cmd = f"salloc --account={args.cc_account} --nodes=1 --mem={args.mem}G --cpus-per-task={args.cpu} --time=0-{args.time}:00"
		if args.gpu > 0:
			# eg. --gpus-per-node=v100:1 or --gpus-per-node=1
			cmd += " --gpus-per-node={args.gpu}"
		script_lines.append(cmd)


    # load modules
	script_lines.extend(
		[	
			"#load modules",
			"module load StdEnv/2023",
			f"module load python/{args.python_version}",
			f"module load rdkit/{args.rdkit_version}",
		]
	)
  
	# create venv on compute node
	script_lines.extend(
		[
			'',
			#"# go to script dir",
			#f"cd {os.path.abspath(args.frag_gnn_dp)}",
			f'FRAGNNET_DP={os.path.abspath(args.frag_gnn_dp)}',
			'WORKING_DIR=${SLURM_TMPDIR}/${SLURM_JOB_ID}',
			'# create work dir',
			'mkdir -p ${WORKING_DIR}',
			''
			'# create local env',
			# go to slurm and create virtual env
			'cd  ${WORKING_DIR}',
			'virtualenv FRAG-GNN',
			'source FRAG-GNN/bin/activate',
			'pip install --no-index --upgrade pip',
			'pip install --no-index torch==2.4.1 \\',
			'    torch_geometric==2.6.1 \\',
			'    torch_scatter \\',
			'    pandas \\',
			'    wandb \\',
			'    tqdm \\',
			'    scipy==1.15.2 \\',
			'    joblib==1.3.2 \\',
			'    matplotlib \\',
			'    pytorch_lightning==2.4.0 \\',
			'    dgl==2.4.0 \\',
			'    omegaconf \\',
			'    cython==3.0.12 \\',
			'    h5py',
			"",
			'# intall from whl',
			'cp ${FRAGNNET_DP}/wheels/pyteomics-4.7.5-py3-none-any.whl ${WORKING_DIR}/pyteomics-4.7.5-py3-none-any.whl',
			'pip install pyteomics-4.7.5-py3-none-any.whl',
			''
		]
	)
 
	# copy fragnnet sourfce code and install to venv
	script_lines.extend(
		[	
   			'# copy script code',
			'rsync -av --exclude=".*" --exclude="__pycache__" ${FRAGNNET_DP}/preproc_scripts ${WORKING_DIR}',
			'rsync -av --exclude=".*" --exclude="__pycache__" ${FRAGNNET_DP}/scripts ${WORKING_DIR}',
			'# copy src code - this is needed to avoid issues with multiple install read the same src code',
			'rsync -av --exclude=".*" --exclude="__pycache__" --exclude="*.so" ${FRAGNNET_DP}/src ${WORKING_DIR}',
			'cp ${FRAGNNET_DP}/setup.cfg ${WORKING_DIR}/',
			'cp ${FRAGNNET_DP}/setup.py ${WORKING_DIR}/',
			'cp ${FRAGNNET_DP}/pyproject.toml ${WORKING_DIR}/',
			'# install with out dependencies, assume all dependencies are installed manually',
			'pip install -I --no-deps ${WORKING_DIR}',
			'',
		]
	)
 
	# https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html
	# lightning seems have issues with srun
	# disable DDP in the trainer
	python_line = f"python {args.python_script_fp}"

	custom_config_fp = None
	# default config template
	config_template_fp = os.path.join(args.frag_gnn_dp,"config","template.yml")

	for kwarg in args.script_kwargs:
		assert "=" in kwarg, kwarg
		key, value = kwarg.split("=")
		if len(key) == 1:
			python_line += f" -{key} {value}"
		elif key.startswith('-'):
			python_line += f" {key} {value}"
		else:
			python_line += f" --{key} {value}"

		if key == "c" or key == "custom_fp":
			custom_config_fp = value
		if key == "t" or key == "template_fp":
			config_template_fp = value

	# parse configs	file
	config_keywords = ['spec_fp','mol_fp', 'ann_fp', 'config','config_template','split_dp','magma_dp','frag_dp', 'model_type', 'mces_fp']
	config_d = {'config_template':config_template_fp}
	if custom_config_fp is not None:
		config_d['config'] = custom_config_fp
  
	# parse configs
	for config_fp in [config_template_fp, custom_config_fp]:
		if config_fp is not None:
			with open(config_fp,"r") as config_file:
				config_lines = config_file.readlines()
				for line in config_lines:
					for config_keyword in config_keywords:
						if line.startswith(config_keyword):
							config_d[config_keyword] = line.split(":")[-1].strip()

	# change keywords based model type
	data_keyword = [ 'config', 'config_template', 'spec_fp','mol_fp', 'split_dp']
	if config_d['model_type'] in ['frag_gnn','fragnnet']:
		data_keyword.append('frag_dp')
	elif config_d['model_type'] in ['iceberg_gen','iceberg_inten']:
		data_keyword.append('magma_dp')
	elif config_d['model_type'] == 'graff':
		data_keyword.append('ann_fp')
	elif config_d['model_type'] in ['mces_pretraining']:
		data_keyword.append('mces_fp')
	print(f"> model type: {config_d['model_type']}")
 	# copy data files	
	script_lines.extend(
			[
				'#copy files'
			]
		)

	for name in data_keyword:
		path_value  = config_d[name]
		if path_value.startswith('./'):
			path_value = path_value[2:]
		if name not in config_d:
			print(f"missing {name} in config file")
		if name != 'frag_dp':
				if not os.path.exists(path_value):
					raise FileNotFoundError(f"File not found: {path_value}")
				print(f"> adding cmd to copy {name} file from {path_value} to slurm tmp dir")
				script_lines.extend(
					[f"# copy {name} file from {path_value} to slurm tmp dir",
					f"rsync -av --relative ${{FRAGNNET_DP}}/./{path_value} ${{WORKING_DIR}}"])
		elif name == 'frag_dp':
			# check if frag_dp is a file or dir, if it is file, copy it and unpack
			tar_ext = None
			dag_dp = path_value
			frag_dp = dag_dp.rsplit("/",1)[0]
			for ext in ['.tar','.tar.xz', '.tar.gz', '.tgz']:
				if os.path.isfile(frag_dp+ext):
					tar_ext = ext
					break
			if tar_ext is not None:
				print(f"> adding cmd to copy {frag_dp+tar_ext}")
				script_lines.extend(
	   				[f"# copy {name} file from {frag_dp}{tar_ext} to slurm tmp dir and unpack",
					 f"rsync -av --relative ${{FRAGNNET_DP}}/./{frag_dp+tar_ext} ${{WORKING_DIR}}",
					f"tar -xf ${{WORKING_DIR}}/{frag_dp+tar_ext} -C ${{WORKING_DIR}}/{frag_dp.rsplit('/',1)[0]}",
					])
			elif os.path.isdir(dag_dp):
				print(f"> adding cmd to copy directory {dag_dp}")
				script_lines.extend(
	   				[f"# copy {name} file from {dag_dp} to slurm tmp dir",
					 f"rsync -av --relative ${{FRAGNNET_DP}}/./{dag_dp} ${{WORKING_DIR}}"
					])
			else:
				raise FileNotFoundError(f"File not found: {frag_dp} or {frag_dp+ext}")

    # soft link back to home directy for wandb
	script_lines.extend(
		[
			'# create wandb directory if not exists',
   			'if [ ! -d ${FRAGNNET_DP}/wandb ]; then',
			'	mkdir ${FRAGNNET_DP}/wandb',
			'fi',
			'# soft link to wandb',
   			'cd ${WORKING_DIR}',
			'if [ ! -e wandb ]; then',
			'	ln -s ${FRAGNNET_DP}/wandb wandb',
			'fi',
		]
	)
 
	# add python line
	script_lines.extend(
		[
			'# run python script',
   			'cd ${WORKING_DIR}',
			python_line
		]
	)
   
	if not args.interactive:
		script_lines.extend(
			[
				"",
				"# record ending",
				'echo `date`: Job ${SLURM_JOB_ID} finished running'
			]
		)

		script_str = "\n".join(script_lines)
		assert os.path.isdir(args.script_dp), args.script_dp
		script_fp = os.path.join(args.script_dp,f"{args.job_name}.sh")
		with open(script_fp,"w") as script_file:
			print(script_str,file=script_file)
	else:
		for row in script_lines:
			print(row)


	if not args.no_submit:
		process = subprocess.Popen(['sbatch', script_fp], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
		out, err = process.communicate()
		print(out.strip(), err.strip())

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--script_dp",type=str, default="slurm_scripts")
	parser.add_argument("--job_dp",type=str, default="jobs")
	parser.add_argument("--python_script_fp",type=str,required=True)
	parser.add_argument("--frag_gnn_dp",type=str, help="location of frag-gnn project", required=True)
	parser.add_argument("--mem", type=int, default=128)
	parser.add_argument("--cpu", type=int, default=8)
	parser.add_argument("--gpu", type=int, default=1)
	parser.add_argument("--time", type=int, default=3)
	parser.add_argument("--script_kwargs", type=str, nargs="+", required=False)
	parser.add_argument("--job_name", type=str, required=True, help="name of the job")
	parser.add_argument('--cc_account', default='rrg-dwishart', choices=['rrg-dwishart','rrg-rgreiner','def-dwishart','def-rgreiner','aip-rgreiner'])
	parser.add_argument('--cc_email', help='email', default='fw4@ualberta.ca')
	#parser.add_argument('--python_env', default="./FRAG-GNN-GPU", help="virtualenv location")
	parser.add_argument('--python_version', default="3.11", choices=['3.10','3.11'], help="python version")
	parser.add_argument('--rdkit_version', default="2023.09.5", choices=['2022.03.5','2023.09.5','2024.03.5'], help="rdkit version")
	parser.add_argument('-i', '--interactive', type = bool, default=False, action=argparse.BooleanOptionalAction, help="set to true to print interative command")
	parser.add_argument('--no_submit', type=bool, default=True)
	args = parser.parse_args()
 
	main(args)