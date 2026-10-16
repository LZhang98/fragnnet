import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge multiple spec_df and mol_df datasets')
    
    # Support for multiple processing directories
    parser.add_argument('--proc_dirs', nargs='+', required=True, 
                       help='List of processing directories to merge (space-separated)')
    parser.add_argument('--dataset_names', nargs='+', 
                       help='Optional list of dataset names/prefixes (if not provided, will use directory names)')
    parser.add_argument('--output_dp', help='Path to output directory for merged datasets')
    
    args = parser.parse_args()
    
    
    # Use multi-dataset mode
    proc_dirs = args.proc_dirs
    if args.dataset_names:
        if len(args.dataset_names) != len(proc_dirs):
            print("Error: Number of dataset names must match number of directories")
            exit(1)
        dataset_names = args.dataset_names
    else:
        # Use directory basenames as dataset names
        dataset_names = [Path(d).name for d in proc_dirs]

    print(f"> Processing directories: {proc_dirs}")
    print(f"> Dataset names: {dataset_names}")
    print(f"> Output: {args.output_dp}")
    os.makedirs(args.output_dp, exist_ok=True)
    
    # Load all datasets
    spec_mol_dfs = []
    merged_mol_df =None
    spec_df_columns = None
 
    for i, (proc_dir, dataset_name) in enumerate(zip(proc_dirs, dataset_names)):
        print(f">> Loading dataset {i+1}/{len(proc_dirs)}: {dataset_name}")
        spec_df = pd.read_pickle(os.path.join(proc_dir, 'spec_df.pkl'))
        mol_df = pd.read_pickle(os.path.join(proc_dir, 'mol_df.pkl'))
        
        print(f"   - Loaded {len(spec_df):,} spectra and {len(mol_df):,} molecules")
        
        # track all columns
        spec_df_columns = spec_df.columns if spec_df_columns is None else spec_df_columns.union(spec_df.columns)
        #mol_df_columns = mol_df.columns if mol_df_columns is None else mol_df_columns.union(mol_df.columns)
        
        # do this to save space
        prev_mol_count = len(merged_mol_df) if merged_mol_df is not None else 0
        merged_mol_df = mol_df if merged_mol_df is None else  pd.concat([merged_mol_df, mol_df], ignore_index=True, join='outer')
        before_dedup = len(merged_mol_df)
        merged_mol_df = merged_mol_df.drop_duplicates(subset=['inchikey_s'])
        after_dedup = len(merged_mol_df)
        
        if i > 0:  # Only show dedup stats after first dataset
            duplicates_removed = before_dedup - after_dedup
            print(f"   - Merged molecules: {before_dedup:,} → {after_dedup:,} (removed {duplicates_removed:,} duplicates)")
        else:
            print(f"   - Base dataset molecules: {after_dedup:,}")

        mol_df = mol_df.drop(columns=['mol'])
        # we only need inchikey_s for id molecule
        spec_mol_df = spec_df.merge(mol_df[['mol_id','inchikey_s']], on='mol_id', how='left')
        spec_mol_df['origin_dataset'] = dataset_name
        spec_mol_df['origin_dataset_id'] = spec_df['spec_id']
        spec_mol_df['mol_id'] = spec_mol_df['inchikey_s']  # temporarily set mol_id to inchikey_s to avoid conflicts
        spec_mol_dfs.append(spec_mol_df)
        #print(spec_mol_df.head(10))

    if merged_mol_df is not None and len(spec_mol_dfs) > 0:
        # in case some datasets don't have all columns, ensure all columns are present
        merged_spec_mol_df = pd.concat(spec_mol_dfs, ignore_index=True, join='outer')
        merged_spec_mol_df = merged_spec_mol_df.reset_index(drop=True)
        merged_spec_mol_df['spec_id'] = np.arange(len(merged_spec_mol_df))
    
        # Reassign mol_id in both tables from one global inchikey_s mapping
        merged_mol_df = merged_mol_df.reset_index(drop=True)
        merged_mol_df['mol_id'] = np.arange(len(merged_mol_df))
        inchikey_to_mol_id = dict(zip(merged_mol_df['inchikey_s'], merged_mol_df['mol_id']))
        merged_spec_mol_df['mol_id'] = merged_spec_mol_df['inchikey_s'].map(inchikey_to_mol_id)

        # Rebuild group_id from global identity (same key as 02_prepare_proc.py)
        merged_spec_mol_df['group_id'] = merged_spec_mol_df.groupby(
            ['mol_id', 'prec_type', 'inst_type', 'frag_mode'], sort=True, dropna=False
        ).ngroup()

        # Save merged dataframes
        merged_spec_df = merged_spec_mol_df[spec_df_columns]
        merged_spec_df.to_pickle(os.path.join(args.output_dp, 'spec_df.pkl'))
        merged_mol_df.to_pickle(os.path.join(args.output_dp, 'mol_df.pkl'))
        
        # Print detailed merge statistics
        print("\n" + "="*60)
        print("MERGE STATISTICS")
        print("="*60)
        
        # Dataset-specific stats
        print("\nDataset Contributions:")
        dataset_stats = merged_spec_mol_df.groupby('origin_dataset').agg({
            'spec_id': 'count',
            'mol_id': 'nunique'
        }).rename(columns={'spec_id': 'spectra_count', 'mol_id': 'unique_molecules'})
        
        for dataset in dataset_stats.index:
            spectra_count = dataset_stats.loc[dataset, 'spectra_count']
            mol_count = dataset_stats.loc[dataset, 'unique_molecules']
            print(f"  {dataset:<15}: {spectra_count:>7,} spectra, {mol_count:>6,} unique molecules")
        
        # Overall merge stats
        print(f"\nMerge Summary:")
        print(f"  Total input datasets:     {len(proc_dirs)}")
        print(f"  Total spectra merged:     {len(merged_spec_df):,}")
        print(f"  Total unique molecules:   {len(merged_mol_df):,}")
        print(f"  Avg spectra per molecule: {len(merged_spec_df) / len(merged_mol_df):.2f}")
        
        # Deduplication stats
        total_input_molecules = sum([len(pd.read_pickle(os.path.join(proc_dir, 'mol_df.pkl'))) 
                                    for proc_dir in proc_dirs])
        molecules_deduplicated = total_input_molecules - len(merged_mol_df)
        print(f"  Input molecules (total):  {total_input_molecules:,}")
        print(f"  Molecules deduplicated:   {molecules_deduplicated:,} ({molecules_deduplicated/total_input_molecules*100:.1f}%)")
        
        # Column information
        print(f"\nDataframe Details:")
        print(f"  spec_df columns:          {len(merged_spec_df.columns)}")
        print(f"  mol_df columns:           {len(merged_mol_df.columns)}")
                        
        # File sizes
        spec_file_path = os.path.join(args.output_dp, 'spec_df.pkl')
        mol_file_path = os.path.join(args.output_dp, 'mol_df.pkl')
        spec_file_size = os.path.getsize(spec_file_path) / (1024*1024)  # MB
        mol_file_size = os.path.getsize(mol_file_path) / (1024*1024)    # MB
        
        print(f"\nOutput Files:")
        print(f"  spec_df.pkl: {spec_file_size:.1f} MB")
        print(f"  mol_df.pkl:  {mol_file_size:.1f} MB")
        print(f"  Total size:  {spec_file_size + mol_file_size:.1f} MB")
        
        print("\n" + "="*60)
        print("MERGE COMPLETED SUCCESSFULLY")
        print("="*60)
        #merged_mol_df = merged_mol_df
    else:
        print("No data to merge.")