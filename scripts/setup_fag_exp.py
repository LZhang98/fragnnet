
import argparse

"""_summary_
som quick code 
"""

if __name__ == "__main__":

	script_lines = []
	script_lines.append("#!/bin/bash")

	max_depths = [2,3,4]

	#tolerances = ["0.01","0.005","10ppm","5ppm"]
	isotopes = [True, False]

	for max_depth in max_depths:
		for isotope in isotopes:
			run_name = f"d{max_depth}_iso{isotope}"
			#for tolerance in tolerances:
			wandb_run_name = f"d{max_depth}_iso{isotope}"
			cmd = f"python ./preproc_scripts/03_prepare_dag_feats.py  --max_depth={max_depth} --isotope={isotope} --wandb_mode=online --wandb_run_name='{wandb_run_name}' --run_name='{run_name}'"
			script_lines.append(cmd)
	#print('\n'.join(script_lines))

	with open("run_03.sh", "w+") as fileout:
		fileout.write('\n'.join(script_lines))