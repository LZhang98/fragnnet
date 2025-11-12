"""
Quick test to verify D-MPNN works with your existing edge format.
Run this to confirm compatibility before full training.
"""
import torch as th
import torch_geometric as pyg
from fragnnet.model.dmpnn import build_dmpnn
from fragnnet.utils.feat_utils import get_mol_graph
from rdkit import Chem

def test_dmpnn_with_fraggnn_format():
    """Test D-MPNN with FragGNNet's edge format"""
    
    # Test molecule
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    
    # Test both bigraph settings
    for bigraph in [False, True]:
        print(f"\n{'='*60}")
        print(f"Testing with pyg_bigraph={bigraph}")
        print(f"{'='*60}")
        
        # Get molecular graph in FragGNNet format
        mol_pyg = get_mol_graph(
            mol=mol,
            atom_feats=["a_onehot", "a_degree"],
            bond_feats=["b_degree"],
            pe_embed_k=0,
            bigraph=bigraph
        )
        
        print(f"Input edges shape: {mol_pyg.edge_index.shape}")
        print(f"Input edge_attr shape: {mol_pyg.edge_attr.shape}")
        print(f"Num nodes: {mol_pyg.x.shape[0]}")
        
        # Create D-MPNN encoder
        dmpnn = build_dmpnn(
            node_feats_size=mol_pyg.x.shape[1],
            edge_feats_size=mol_pyg.edge_attr.shape[1],
            hidden_size=64,
            num_layers=2,
            dropout=0.0,
            optimized=True
        )
        
        # Forward pass
        try:
            batch = th.zeros(mol_pyg.x.shape[0], dtype=th.long)
            node_embeddings = dmpnn(
                x=mol_pyg.x,
                edge_index=mol_pyg.edge_index,
                edge_attr=mol_pyg.edge_attr,
                batch_index=batch
            )
            
            print(f"✅ SUCCESS!")
            print(f"Output shape: {node_embeddings.shape}")
            print(f"Output dtype: {node_embeddings.dtype}")
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            raise
    
    print(f"\n{'='*60}")
    print(f"✅ All tests passed! D-MPNN works with your edge format.")
    print(f"{'='*60}")
    
    # Recommendation
    print("\n📝 RECOMMENDATION:")
    print("   Use pyg_bigraph: False with D-MPNN")
    print("   (D-MPNN creates bidirectional edges internally)")
    
if __name__ == "__main__":
    test_dmpnn_with_fraggnn_format()
