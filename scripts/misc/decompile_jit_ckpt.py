from frag_gnn.utils.nn_utils import decompile_jit_ckpt

if __name__ == "__main__":
    
    #decompile_jit_ckpt('saved_ckpts/icml_inchikey/massformer/massformer_inchikey.ckpt', 'saved_ckpts/icml_inchikey/massformer/massformer_inchikey_nocp.ckpt')
    #decompile_jit_ckpt('saved_ckpts/icml_inchikey/neims/neims_inchikey.ckpt', 'saved_ckpts/icml_inchikey/neims/neims_inchikey_nocp.ckpt')
    #decompile_jit_ckpt('saved_ckpts/icml_scaffold/massformer/massformer_scaffold.ckpt', 'saved_ckpts/icml_scaffold/massformer/massformer_scaffold_nocp.ckpt')
    #decompile_jit_ckpt('saved_ckpts/icml_scaffold/neims/neims_scaffold.ckpt', 'saved_ckpts/icml_scaffold/neims/neims_scaffold_nocp.ckpt')
    decompile_jit_ckpt('saved_ckpts/retrieval/massformer_v2_inchikey.ckpt', 'saved_ckpts/retrieval/massformer_v2_inchikey_nocp.ckpt')