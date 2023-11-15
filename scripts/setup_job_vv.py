import os
import argparse
from distutils.util import strtobool

def booltype(x):
	return bool(strtobool(x))

QOS_TO_TIME = {
	"normal": 16,
	"m": 12,
	"m2": 8,
	"m3": 4,
	"long": 48,
	"high": 8,
	"affiliate": 8,
	"deadline": 7*24, # big number
	"cpu_qos": 7*24 # big number
}

QOS_TO_GPU = {
	"normal": 2,
	"m": 4,
	"m2": 8,
	"m3": 16,
	"long": 1,
	"high": 4,
	"affiliate": 2,
	"deadline": 12,
	"cpu_qos": 0
}

def time_check(time, qos):

	assert time > 0, time
	assert time <= QOS_TO_TIME[qos], (time, qos, QOS_TO_TIME[qos])

def gpu_check(gpu, qos):

	assert gpu <= QOS_TO_GPU[qos], (gpu, qos, QOS_TO_GPU[qos])

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
	if args.time == -1:
		args.time = QOS_TO_TIME[args.qos]
	time_check(args.time, args.qos)
	script_lines.append(f"#SBATCH --time=0-{args.time-1}:59")
	if args.gpu > 0:
		assert args.qos != "cpu_qos", args.qos
		gpu_check(args.gpu, args.qos)
		script_lines.append(f"#SBATCH --gres=gpu:{args.gpu_type}:{args.gpu}")
	if args.qos == "deadline":
		script_lines.append("#SBATCH --account=deadline")
	elif args.qos == "cpu_qos":
		script_lines.append("#SBATCH --partition=cpu")
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
	if args.requeue:
		python_line = f"timeout -k 2m {args.time*60-5}m " + python_line
		script_lines.append(python_line)
		script_lines.extend(
			[
				"if [[ $? == 124 ]]; then",
				'    echo "Timed out, Requeuing"',
				"    scontrol requeue ${SLURM_JOB_ID}",
				"fi"
			]
		)
	else:
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
	parser.add_argument("--time", type=int, default=-1, help="time in hours (-1 means use default)")
	parser.add_argument("--gpu_type", type=str, default="t4", choices=["t4","rtx6000","a40"])
	parser.add_argument("--qos", type=str, default="normal", choices=["normal","m","m2","m3","long","high","deadline","affiliate","cpu_qos"])
	parser.add_argument("--env", type=str, default="FRAG-GNN-GPU3")
	parser.add_argument("--script_kwargs", type=str, nargs="+", required=False)
	parser.add_argument("--job_name", type=str, required=True)
	parser.add_argument("--job_id", type=int, required=False, help="job_id (-1 means use the slurm job id)")
	parser.add_argument("--blacklist", type=str, nargs="+", required=False)
	parser.add_argument("--modules", type=str, nargs="+", default=["cuda-11.8"])
	parser.add_argument("--home_dir", type=str, default="/fs01/home/adamo")
	parser.add_argument("--requeue", type=booltype, default=False)
	args = parser.parse_args()
	main(args)