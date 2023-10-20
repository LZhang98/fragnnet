import os
import argparse

from frag_gnn.utils.misc_utils import booltype

def time_check(time, qos):

	assert time > 0, time
	if qos == "normal":
		assert time <= 16, time
	elif qos == "m":
		assert time <= 12, time
	elif qos == ["m2", "high", "affiliate"]:
		assert time <= 8, time
	elif qos == "m3":
		assert time <= 4, time
	elif qos == "long":
		assert time <= 48, time

def gpu_check(gpu, qos):

	if qos in ["normal", "affiliate"]:
		assert gpu <= 2, gpu
	elif qos in ["m", "high"]:
		assert gpu <= 4, gpu
	elif qos == "m2":
		assert gpu <= 8, gpu
	elif qos == "m3":
		assert gpu <= 16, gpu
	elif qos == "long":
		assert gpu <= 1, gpu
	elif qos == "deadline":
		assert gpu <= 12, gpu

def main(args):

	script_lines = []
	script_lines.extend(
		[
			"#!/bin/bash",
			"#SBATCH -N 1",
			"#SBATCH -n 1"
		]
	)
	script_lines.append(f"#SBATCH --job-name={args.job_name}")
	script_lines.append(f"#SBATCH --mem={args.mem}G")
	script_lines.append(f"#SBATCH --cpus-per-task={args.cpu}")
	time_check(args.time, args.qos)
	script_lines.append(f"#SBATCH --time=0-{args.time-1}:59")
	if args.gpu > 0:
		gpu_check(args.gpu, args.qos)
		script_lines.append(f"#SBATCH --gres=gpu:{args.gpu_type}:{args.gpu}")
	if args.qos == "deadline":
		script_lines.append("#SBATCH --account=deadline")
	script_lines.append(f"#SBATCH --qos={args.qos}")
	script_lines.extend(
		[
			f"#SBATCH --output={args.home_dir}/jobs/%j.out",
			f"#SBATCH --error={args.home_dir}/jobs/%j.err",
			"#SBATCH --open-mode=append"
		]
	)
	if args.blacklist is not None:
		for bl_node in args.blacklist:
			script_lines.append(f"#SBATCH --exclude={bl_node}")
	script_lines.extend(
		[
			"",
			"# record start",
			"echo `date`: Job ${SLURM_JOB_ID} is allocated resources"
		]
	)
	script_lines.extend(
		[
			"",
			"# make wandb directory (if necessary)",
			"mkdir -p /checkpoint/adamo/wandb",
			"touch /checkpoint/adamo/wandb/DELAYPURGE",
			f"ln -sfn /checkpoint/adamo/wandb {args.home_dir}/frag-gnn/wandb"
		]
	)
	script_lines.extend(
		[
			"",
			"# source .bashrc",
			f". {args.home_dir}/miniconda3/etc/profile.d/conda.sh"
		]
	)
	if args.modules is not None:
		script_lines.extend(
			[
				"",
				"# load modules",
			]
		)
		for module in args.modules:
			script_lines.append(f"module load {module}")
	script_lines.extend(
		[
			"",
			"# activate environment",
			f"conda activate {args.env}"
		]
	)
	script_lines.extend(
		[
			"",
			"# run script",
			f"cd {args.home_dir}/frag-gnn"
		]
	)
	python_line = f"python {args.python_script_fp}"
	script_kwargs = args.script_kwargs
	for kwarg in script_kwargs:
		assert "=" in kwarg, kwarg
		key, value = kwarg.split("=")
		python_line += f" --{key} {value}"
	if args.job_id:
		if args.job_id == -1:
			# default
			python_line += " --job_id ${SLURM_JOB_ID}"
		else:
			python_line += f" --job_id {args.job_id}"
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
	parser.add_argument("--python_script_fp",type=str,required=True)
	parser.add_argument("--mem", type=int, default=64)
	parser.add_argument("--cpu", type=int, default=8)
	parser.add_argument("--gpu", type=int, default=1)
	parser.add_argument("--time", type=int, default=16)
	parser.add_argument("--gpu_type", type=str, default="t4", choices=["t4","rtx6000","a40"])
	parser.add_argument("--qos", type=str, default="normal", choices=["normal","m","m2","m3","long","high","deadline","affiliate"])
	parser.add_argument("--env", type=str, default="FRAG-GNN-GPU3")
	parser.add_argument("--script_kwargs", type=str, nargs="+", required=False)
	parser.add_argument("--job_name", type=str, required=True)
	parser.add_argument("--job_id", type=int, required=False)
	parser.add_argument("--blacklist", type=str, nargs="+", required=False)
	parser.add_argument("--modules", type=str, nargs="+", default=["cuda-11.8"])
	parser.add_argument("--home_dir", type=str, default="/fs01/home/adamo")
	args = parser.parse_args()
	main(args)