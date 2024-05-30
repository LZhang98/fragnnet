import yaml
import os



if __name__ == '__main__': 
    model_names = ['massformer', 'neims', 'fraggnn_d3', 'fraggnn_d4', 'massformer_v2','neims_v2', 'fraggnn_d3_v2', 'fraggnn_d4_v2']
    split_name = 'nist20_d4_wr00_inchikey'
    for model_name in model_names:
        for mnce_b in [True, False]:
            src_config_fp = f'./config/{split_name}/{model_name}/s0.yml'

                #print(src_config)
            dest_config_dp = f'./config/deepmet_cvs/{split_name}/{model_name}' if mnce_b else f'./config/deepmet_cvs/{split_name}/{model_name}_nomnce'
            
            print(f"> src config {src_config_fp}")
            print(f"> model name {model_name}")
            print(f"> create {dest_config_dp}")
            
            os.makedirs(dest_config_dp, exist_ok=True)
            for cv_idx in range(11):
                with open(src_config_fp, 'r') as file:
                    src_config = yaml.safe_load(file)
                    cv_config = src_config.copy()
                cv_config['wandb_tags'].append(f'deepmet')
                cv_config['wandb_tags'].append(f'cv{cv_idx}')
                #cv_config['wandb_tags'].append('no_compile')
                cv_config['wandb_group']= f'{split_name}_{model_name}_deepmet_cv'  if mnce_b == False else f'{split_name}_{model_name}_nomnce_deepmet_cv'
                cv_config['wandb_name'] = f'{split_name}_{model_name}_deepmet_cv{cv_idx}'  if mnce_b == False else f'{split_name}_{model_name}_nomnce_deepmet_cv{cv_idx}'
                cv_config['split_dp']= f'data/split/cv_deepmet2/cv_{cv_idx}'
                cv_config['spec_params']['merge_keep_ces'] = mnce_b
                cv_config['spec_params']['nce'] = mnce_b
                if mnce_b == False:
                    cv_config['m_ce_insert_mlp_embed'] = False
                    cv_config['m_ce_insert_mlp_bin'] = False
                    cv_config['wandb_tags'].remove('m_nce_embed')
                elif 'm_nce_embed' not in cv_config['wandb_tags']:
                    cv_config['wandb_tags'].append('m_nce_embed')
                    
                cv_config['compile'] = True
                # don't delete checkpoints
                cv_config['delete_checkpoints'] = False
                with open(os.path.join(dest_config_dp, f"cv_{cv_idx}.yml"), 'w') as outfile:
                    yaml.dump(cv_config, outfile, default_flow_style=False, sort_keys=False)
                
