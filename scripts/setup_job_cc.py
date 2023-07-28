import os
import argparse

def main(args):

	# create folders
	os.makedirs(f"{os.path.abspath(args.frag_gnn_dp)}/wandb", exist_ok=True)
	os.makedirs(f"{os.path.abspath(args.script_dp)}", exist_ok=True)
	os.makedirs(f"{os.path.abspath(args.job_dp)}", exist_ok=True)

	script_lines = []
	# account, ram, cpu, gpu, and etc
	script_lines.extend(
		[
			"#!/bin/bash",
			f"#SBATCH --account={args.cc_account}",			
			"#SBATCH --nodes=1",
			f"#SBATCH --job-name={args.job_name}",
			f"#SBATCH --mem={args.mem}G",
			f"#SBATCH --cpus-per-task={args.cpu}",
			f"#SBATCH --time=0-{args.time}:00",
			f"#SBATCH --mail-user={args.cc_email}",
			f"#SBATCH --mail-type=ALL"
		]
	)

	# config where to save job output and job error
	# append to existing onces
	script_lines.extend(
		[
			f"#SBATCH --output={os.path.abspath(args.job_dp)}/%j.out",
			f"#SBATCH --error={os.path.abspath(args.job_dp)}/%j.err",
			"#SBATCH --open-mode=append"
		]
	)
	# add gpus
	if args.gpu > 0:
		script_lines.append(f"#SBATCH --gres=gpu:{args.gpu}")

	# start recording 	
	script_lines.extend(
		[
			"", 
			"# record start",
			"echo `date`: Job ${SLURM_JOB_ID} is allocated resources"
		]
	)
	
    # load modules
	script_lines.extend(
		[
			"module load cuda/11.7",
			"module load module load python/3.10",
		]
	)
  
	# for now let us just using existing env
	# go to working dir
	# use abs dir
	script_lines.extend(
		[
			"",
			"# run script",
			f"cd {os.path.abspath(args.frag_gnn_dp)}",
			f"source {args.python_env}/bin/activate"
		]
	)
 
	# and not do copy
	python_line = f"python {args.python_script_fp}"
	#print(args.script_kwargs)
	if args.script_kwargs not in [None, ""]:
		script_kwargs = args.script_kwargs.split(",")
		for kwarg in script_kwargs:
			assert "=" in kwarg, kwarg
			key, value = kwarg.split("=")
			python_line += f" --{key} {value}"
	script_lines.append(python_line)
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


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--script_dp",type=str, default="slurm_scripts")
	parser.add_argument("--job_dp",type=str, default="jobs")
	parser.add_argument("--python_script_fp",type=str,required=True)
	parser.add_argument("--frag_gnn_dp",type=str, help="location of frag-gnn project", required=True)
	parser.add_argument("--mem", type=int, default=64)
	parser.add_argument("--cpu", type=int, default=8)
	parser.add_argument("--gpu", type=int, default=1)
	parser.add_argument("--time", type=int, default=12)
	parser.add_argument("--script_kwargs", type=str, help='arugments for python script eg."-w=online,-c=config/debug_d4_tiny.yml"', default="")
	parser.add_argument("--job_name", type=str, required=True)
	parser.add_argument('--cc_email', help='email', default='fw4@ualberta.ca')
	parser.add_argument('--cc_account', default='rrg-dwishart', choices=['rrg-dwishart'])
	parser.add_argument('--python_env', default="./FRAG-GNN-GPU2", help="virtualenv location")
	args = parser.parse_args()
	main(args)