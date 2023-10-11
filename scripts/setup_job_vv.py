import os
import argparse

from frag_gnn.utils.misc_utils import booltype

def main(args):

	home_dir = "/ssd005/home/adamo"
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
	assert args.time > 0, args.time
	script_lines.append(f"#SBATCH --time=0-{args.time-1}:59")
	if args.partition != "cpu":
		assert args.gpu > 0, args.gpu
		script_lines.append(f"#SBATCH --gres=gpu:{args.gpu}")
	else:
		assert args.qos == "nopreemption", args.qos
	script_lines.append(f"#SBATCH --partition={args.partition}")
	if args.qos == "deadline":
		script_lines.append("#SBATCH --account=deadline")
	script_lines.append(f"#SBATCH --qos={args.qos}")
	script_lines.extend(
		[
			f"#SBATCH --output={home_dir}/jobs/%j.out",
			f"#SBATCH --error={home_dir}/jobs/%j.err",
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
			f"ln -sfn /checkpoint/adamo/wandb {home_dir}/frag-gnn/wandb"
		]
	)
	script_lines.extend(
		[
			"",
			"# source .bashrc",
			f". {home_dir}/miniconda3/etc/profile.d/conda.sh"
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
			f"cd {home_dir}/frag-gnn"
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
	parser.add_argument("--time", type=int, default=24)
	parser.add_argument("--partition", type=str, default="cpu", choices=["t4v2","rtx6000","interactive","cpu","a40"])
	parser.add_argument("--qos", type=str, default="nopreemption", choices=["normal","high","nopreemption","deadline"])
	parser.add_argument("--env", type=str, default="FRAG-GNN-GPU2")
	parser.add_argument("--script_kwargs", type=str, nargs="+", required=False)
	parser.add_argument("--job_name", type=str, required=True)
	parser.add_argument("--job_id", type=int, required=False)
	parser.add_argument("--blacklist", type=str, nargs="+", required=False)
	parser.add_argument("--modules", type=str, nargs="+", required=False)
	args = parser.parse_args()
	main(args)