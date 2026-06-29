import h5py

if __name__ == "__main__":
    h5_fp = "./data/mces-data/nist20_inputs/mces_input_1.h5"
    with h5py.File(h5_fp, "r") as f:
        print("Groups/Datasets in file:")

        def print_h5(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")

        f.visititems(print_h5)

        # Example: Show first 5 items of each dataset
        for name, ds in f.items():
            if isinstance(ds, h5py.Dataset):
                print(f"\nFirst 5 entries in {name}:")
                print(ds[:5])

        # Check if computation_indices and smiles exist and validate indices
        if "computation_indices" in f and "smiles" in f:
            comp_indices_ds = f["computation_indices"]
            smiles_ds = f["smiles"]

            # Make sure they are datasets
            if isinstance(comp_indices_ds, h5py.Dataset) and isinstance(smiles_ds, h5py.Dataset):
                computation_indices = comp_indices_ds[:]
                smiles = smiles_ds[:]

                print("\nValidation check:")
                print(f"Size of smiles: {len(smiles)}")
                print(f"Size of computation_indices: {len(computation_indices)}")
                print(f"Shape of computation_indices: {computation_indices.shape}")

                if len(computation_indices) > 0:
                    # Check columns 1 and 2 (not column 0 which is the index)
                    if computation_indices.ndim > 1 and computation_indices.shape[1] >= 3:
                        col1_values = computation_indices[:, 1]
                        col2_values = computation_indices[:, 2]

                        min_col1 = int(col1_values.min()) if len(col1_values) > 0 else -1
                        max_col1 = int(col1_values.max()) if len(col1_values) > 0 else -1
                        min_col2 = int(col2_values.min()) if len(col2_values) > 0 else -1
                        max_col2 = int(col2_values.max()) if len(col2_values) > 0 else -1

                        print(f"Column 1 (smiles_1 index) range: {min_col1} to {max_col1}")
                        print(f"Column 2 (smiles_2 index) range: {min_col2} to {max_col2}")

                        # Check if values are within bounds (0-based indexing: should be >= 0 and < len(smiles))
                        errors = []
                        if min_col1 < 0:
                            errors.append(f"Column 1 (smiles_1) min ({min_col1}) < 0")
                        if min_col2 < 0:
                            errors.append(f"Column 2 (smiles_2) min ({min_col2}) < 0")
                        if max_col1 >= len(smiles):
                            errors.append(
                                f"Column 1 (smiles_1) max ({max_col1}) >= size of smiles ({len(smiles)})"
                            )
                        if max_col2 >= len(smiles):
                            errors.append(
                                f"Column 2 (smiles_2) max ({max_col2}) >= size of smiles ({len(smiles)})"
                            )

                        if errors:
                            print("❌ ERROR: Index out of bounds!")
                            for error in errors:
                                print(f"   {error}")
                            print("   This will cause index out of bounds errors!")
                        else:
                            print("✅ OK: All values in columns 1 and 2 are within bounds")
                            print(
                                f"   Column 1 (smiles_1) range: {min_col1} to {max_col1} (valid: 0 to {len(smiles) - 1})"
                            )
                            print(
                                f"   Column 2 (smiles_2) range: {min_col2} to {max_col2} (valid: 0 to {len(smiles) - 1})"
                            )
                    else:
                        print(
                            "❌ ERROR: computation_indices doesn't have expected shape (need at least 3 columns)"
                        )
                        print(f"   Current shape: {computation_indices.shape}")
                else:
                    print("❌ ERROR: computation_indices is empty")
            else:
                print("❌ ERROR: computation_indices or smiles are not datasets")
        else:
            missing = []
            if "computation_indices" not in f:
                missing.append("computation_indices")
            if "smiles" not in f:
                missing.append("smiles")
            print(f"\n❌ ERROR: Missing datasets: {missing}")
