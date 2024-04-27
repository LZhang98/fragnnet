
import pandas as pd
from tqdm import tqdm
import argparse
import os

if __name__ == "__main__":
	
	base_dir = 'data/benchmark/icml_ms2c'
	for split in ['inchikey','scaffold','random_seed']:
		split_dir = os.path.join(base_dir,split)
		stats_rows = []
		csv_names = [f for f in os.listdir(split_dir) if not f.startswith("overall")]
		csv_names.sort()
		for csv_name in csv_names:
			print(csv_name)
			model_name = csv_name.rsplit('_',1)[0]
			df = pd.read_csv(os.path.join(split_dir, csv_name))
			
			ranking_methods = ['cos_sim_hun','cos_sim_bin']
			for ranking_method in ranking_methods:
				stats_rows.append([f'{model_name}_{ranking_method}'])
   
				#print(ranking_method)
				for k in [1,2,3,5,10]:
					#print(k, len(df[df[ranking_method] <= k]), len(df[df[ranking_method] <= k])/ len(df) * 100)
					stats_rows[-1].append(len(df[df[ranking_method] <= k]))
					stats_rows[-1].append(len(df[df[ranking_method] <= k])/ len(df) * 100)

		output_fp = os.path.join(split_dir,f'overall_stats_{split}.csv')
		output_df = pd.DataFrame(stats_rows, columns=['name', 'top-1', 'top-1(%)',
                                                	'top-2','top-2(%)',
													'top-3','top-3(%)',
													'top-5','top-5(%)',
													'top-10','top-10(%)',])
		print(output_df)
		print(output_fp)
		output_df.to_csv(output_fp, index=False)