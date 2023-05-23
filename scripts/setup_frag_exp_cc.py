import argparse
import os

from frag_gnn.utils.misc_utils import booltype

def main(args):

	job_name = f"frag_exp_{args.job_name}"

	frag_df_name = f"frag_df_{args.job_name}.pkl"

	python_line = f"python frag_exp.py"
	python_line += f" --max_depth {args.max_depth}"
	python_line += f" --max_time {args.max_time}"
	python_line += f" --h_prior {args.h_prior}"
	python_line += f" --max_h_transfer {args.max_h_transfer}"
	python_line += f" --frag_df_name {frag_df_name}"
	python_line += f" --allowed_elements {args.allowed_elements}"
	python_line += f" --tolerance {args.tolerance}"
	python_line += f" --isotopes {args.isotopes}"
	python_line += f" --load_frag_df False"
	python_line += f" --wandb_mode {args.wandb_mode}"
	python_line += f" --run_name {job_name}"
	python_line += " --job_id ${SLURM_JOB_ID}"
	python_line += f" --project_dp {args.project_dp}"

	script_lines = []
	script_lines.append("#!/bin/bash")
	script_lines.append("#SBATCH --ntasks=1")
	script_lines.append("#SBATCH --nodes=1")
	script_lines.append(f"#SBATCH --account={args.allocation}-hroest")
	script_lines.append(f"#SBATCH --mem={args.mem*1000}M")
	script_lines.append(f"#SBATCH --time=0-{args.hours:02d}:00")
	script_lines.append(f"#SBATCH --cpus-per-task={args.cpus}")
	script_lines.append(f"#SBATCH --job-name={job_name}")
	script_lines.append("#SBATCH --output=/home/youngad2/projects/def-hroest/youngad2/jobs/%j.out")
	script_lines.append("#SBATCH --error=/home/youngad2/projects/def-hroest/youngad2/jobs/%j.err")
	script_lines.append("#SBATCH --open-mode=append")
	script_lines.append("")
	script_lines.append("set -e")
	script_lines.append("")
	script_lines.append(". /home/youngad2/.bashrc")
	script_lines.append("")
	script_lines.append("module load StdEnv/2020 gcc/9.3.0 python/3.10 rdkit/2022.09.4 cuda/11.1.1 rust/1.65.0")
	script_lines.append("")
	script_lines.append("source /home/youngad2/projects/rrg-hroest/youngad2/GFN/bin/activate")
	script_lines.append("")
	script_lines.append("cd /home/youngad2/projects/def-hroest/youngad2/frag-gnn/frag")
	script_lines.append("")
	script_lines.append(python_line)
	script_lines.append("")
	script_lines.append("ln -s /home/youngad2/projects/def-hroest/youngad2/jobs/${SLURM_JOB_ID}.out " + f"/home/youngad2/projects/def-hroest/youngad2/frag-gnn/jobs/{job_name}.out")
	script_lines.append("")

	script_str = "\n".join(script_lines)
	os.makedirs(args.script_dp,exist_ok=True)
	script_fp = os.path.join(args.script_dp,job_name+".sh")
	with open(script_fp,"w") as script_file:
		print(script_str,file=script_file)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--max_depth",type=int,default=4)
	parser.add_argument("--max_time",type=int,default=3*60)
	parser.add_argument("--h_prior",type=booltype,default=True)
	parser.add_argument("--max_h_transfer",type=int,default=4)
	parser.add_argument("--mem",type=int,default=250)
	parser.add_argument("--cpus",type=int,default=8)
	parser.add_argument("--hours",type=int,default=12)
	parser.add_argument("--script_dp",type=str,default="slurm_scripts")
	parser.add_argument("--allocation", type=str, default="rrg", choices=["def","rrg"])
	parser.add_argument("--allowed_elements",type=str,default="C,H,O,N,F,S,P,Cl")
	parser.add_argument("--tolerance",type=float,default=0.01)
	parser.add_argument("--isotopes",type=booltype,default=True)
	parser.add_argument("--job_name",type=str,default="debug")
	parser.add_argument("--wandb_mode",type=str,default="offline",choices=["off","online","offline"])
	parser.add_argument("--project_dp",type=str,default="/home/youngad2/projects/def-hroest/youngad2/frag-gnn")
	args = parser.parse_args()

	main(args)