from rdkit import Chem

def extract_substructure(mol, atom_ids):
    # Create an empty molecule and add only the atoms of interest
    new_mol = Chem.RWMol()

    # Add atoms that are in the atom_ids list to the new molecule
    atom_map = {}
    for atom_idx in atom_ids:
        atom = mol.GetAtomWithIdx(atom_idx)
        new_atom_idx = new_mol.AddAtom(atom)
        atom_map[atom_idx] = new_atom_idx

    # Now we add the bonds that correspond to the atoms in atom_ids
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()

        # Check if both atoms are in the atom_ids list, if so, add the bond
        if idx1 in atom_ids and idx2 in atom_ids:
            new_mol.AddBond(atom_map[idx1], atom_map[idx2], bond.GetBondType())

    # Get the substructure as a SMILES string
    substructure_smiles = Chem.MolToSmiles(new_mol, isomericSmiles=True)
    return substructure_smiles

def match_substructure(target_smiles, substructure_smiles):
    target_mol = Chem.MolFromSmiles(target_smiles)
    substructure_mol = Chem.MolFromSmiles(substructure_smiles)
    
    # Perform substructure match
    matches = target_mol.GetSubstructMatches(substructure_mol)
    return matches

# Example usage
target_smiles = 'CC(C)CC1=CC=C(C=C1)C(C)C'  # Target molecule
mol = Chem.MolFromSmiles(target_smiles)

# List of atom indices to form the substructure
atom_ids = [0, 1, 2, 5, 6]

# Extract substructure from atom indices
substructure_smiles = extract_substructure(mol, atom_ids)
print("Substructure SMILES:", substructure_smiles)

# Perform substructure match with the target molecule
matches = match_substructure(target_smiles, substructure_smiles)

print("Substructure Matches:", matches)
