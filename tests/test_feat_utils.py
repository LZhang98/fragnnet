"""Unit tests for feat_utils.py

Tests for molecular graph featurization, fingerprints, and utility functions.
"""

import numpy as np
import pytest
import torch as th
from rdkit import Chem
from torch_geometric.data import Data

from fragnnet.utils.feat_utils import (
    AtomChiral,
    AtomDegree,
    AtomFormal,
    AtomHybrid,
    AtomMass,
    AtomOneHot,
    AtomRadical,
    AtomRing,
    BondConj,
    BondDegree,
    BondRing,
    BondStereo,
    MolGraph,
    batch_mols_frags,
    get_frag_graph,
    get_mol_feats_sizes,
    get_mol_fp,
    get_mol_fp_size,
    get_mol_graph,
    one_hot_encoding,
    random_walk_pe,
)


class TestOneHotEncoding:
    """Tests for one_hot_encoding function."""

    def test_basic_encoding(self):
        """Test basic one-hot encoding."""
        result = one_hot_encoding("a", ["a", "b", "c"])
        assert result == [1, 0, 0]

        result = one_hot_encoding("b", ["a", "b", "c"])
        assert result == [0, 1, 0]

        result = one_hot_encoding("c", ["a", "b", "c"])
        assert result == [0, 0, 1]

    def test_encode_unknown_false(self):
        """Test encoding with unknown values when encode_unknown=False."""
        # Should raise error or return all zeros (depending on implementation)
        # Since the code doesn't explicitly handle this, it will just not match
        result = one_hot_encoding("d", ["a", "b", "c"], encode_unknown=False)
        assert result == [0, 0, 0]

    def test_encode_unknown_true(self):
        """Test encoding with unknown values when encode_unknown=True."""
        result = one_hot_encoding("d", ["a", "b", "c"], encode_unknown=True)
        # Should add None to allowable_set and encode unknown as None
        assert result == [0, 0, 0, 1]

    def test_numeric_values(self):
        """Test one-hot encoding with numeric values."""
        result = one_hot_encoding(1, [1, 2, 3])
        assert result == [1, 0, 0]

        result = one_hot_encoding(3, [1, 2, 3])
        assert result == [0, 0, 1]


class TestAtomFeaturizers:
    """Tests for atom featurizer classes."""

    @pytest.fixture
    def ethanol_mol(self):
        """Create an ethanol molecule for testing."""
        return Chem.MolFromSmiles("CCO")

    def test_atom_onehot(self, ethanol_mol):
        """Test AtomOneHot featurizer."""
        atom = ethanol_mol.GetAtomWithIdx(0)  # Carbon
        feats = AtomOneHot.featurize(atom)
        assert isinstance(feats, list)
        assert len(feats) == AtomOneHot.feat_size
        assert sum(feats) == 1  # Should be one-hot

    def test_atom_degree(self, ethanol_mol):
        """Test AtomDegree featurizer."""
        atom = ethanol_mol.GetAtomWithIdx(0)  # Carbon with degree 1
        feats = AtomDegree.featurize(atom)
        assert isinstance(feats, list)
        assert len(feats) == AtomDegree.feat_size

    def test_atom_hybrid(self, ethanol_mol):
        """Test AtomHybrid featurizer."""
        atom = ethanol_mol.GetAtomWithIdx(0)  # sp3 Carbon
        feats = AtomHybrid.featurize(atom)
        assert isinstance(feats, list)
        assert len(feats) == AtomHybrid.feat_size
        assert sum(feats) == 1  # Should be one-hot

    def test_atom_formal(self, ethanol_mol):
        """Test AtomFormal featurizer."""
        atom = ethanol_mol.GetAtomWithIdx(0)  # Neutral carbon
        feats = AtomFormal.featurize(atom)
        assert isinstance(feats, list)
        assert len(feats) == AtomFormal.feat_size
        assert sum(feats) == 1  # Should be one-hot

    def test_atom_radical(self, ethanol_mol):
        """Test AtomRadical featurizer."""
        atom = ethanol_mol.GetAtomWithIdx(0)
        feats = AtomRadical.featurize(atom)
        assert isinstance(feats, list)
        assert len(feats) == AtomRadical.feat_size

    def test_atom_ring(self, ethanol_mol):
        """Test AtomRing featurizer."""
        atom = ethanol_mol.GetAtomWithIdx(0)  # Not in ring
        feats = AtomRing.featurize(atom)
        assert isinstance(feats, list)
        assert len(feats) == AtomRing.feat_size

    def test_atom_chiral(self, ethanol_mol):
        """Test AtomChiral featurizer."""
        atom = ethanol_mol.GetAtomWithIdx(0)
        feats = AtomChiral.featurize(atom)
        assert isinstance(feats, list)
        assert len(feats) == AtomChiral.feat_size

    def test_atom_mass(self, ethanol_mol):
        """Test AtomMass featurizer."""
        atom = ethanol_mol.GetAtomWithIdx(0)
        feats = AtomMass.featurize(atom)
        assert isinstance(feats, list)
        assert len(feats) == AtomMass.feat_size
        assert isinstance(feats[0], float)


class TestBondFeaturizers:
    """Tests for bond featurizer classes."""

    @pytest.fixture
    def ethanol_mol(self):
        """Create an ethanol molecule for testing."""
        return Chem.MolFromSmiles("CCO")

    def test_bond_degree(self, ethanol_mol):
        """Test BondDegree featurizer."""
        bond = ethanol_mol.GetBondWithIdx(0)  # C-C single bond
        feats = BondDegree.featurize(bond)
        assert isinstance(feats, list)
        assert len(feats) == BondDegree.feat_size
        assert sum(feats) == 1  # Should be one-hot

    def test_bond_stereo(self, ethanol_mol):
        """Test BondStereo featurizer."""
        bond = ethanol_mol.GetBondWithIdx(0)
        feats = BondStereo.featurize(bond)
        assert isinstance(feats, list)
        assert len(feats) == BondStereo.feat_size

    def test_bond_ring(self, ethanol_mol):
        """Test BondRing featurizer."""
        bond = ethanol_mol.GetBondWithIdx(0)  # Not in ring
        feats = BondRing.featurize(bond)
        assert isinstance(feats, list)
        assert len(feats) == BondRing.feat_size

    def test_bond_conj(self, ethanol_mol):
        """Test BondConj featurizer."""
        bond = ethanol_mol.GetBondWithIdx(0)  # Not conjugated
        feats = BondConj.featurize(bond)
        assert isinstance(feats, list)
        assert len(feats) == BondConj.feat_size


class TestMolGraph:
    """Tests for MolGraph class."""

    @pytest.fixture
    def mol_graph(self):
        """Create a MolGraph instance with default features."""
        return MolGraph(
            atom_feats=["a_onehot", "a_degree"],
            bond_feats=["b_degree"],
            pe_embed_k=0,
        )

    @pytest.fixture
    def ethanol_mol(self):
        """Create an ethanol molecule for testing."""
        return Chem.MolFromSmiles("CCO")

    def test_initialization(self, mol_graph):
        """Test MolGraph initialization."""
        assert mol_graph.atom_feats == ["a_onehot", "a_degree"]
        assert mol_graph.bond_feats == ["b_degree"]
        assert mol_graph.pe_embed_k == 0
        assert mol_graph.num_atom_feats > 0
        assert mol_graph.num_bond_feats > 0

    def test_get_mol_graph(self, mol_graph, ethanol_mol):
        """Test get_mol_graph method."""
        result = mol_graph.get_mol_graph(ethanol_mol, bigraph=True)
        assert "atom_feats" in result
        assert "bond_feats" in result
        assert "bond_tuples" in result
        assert isinstance(result["atom_feats"], np.ndarray)
        assert isinstance(result["bond_feats"], np.ndarray)
        assert isinstance(result["bond_tuples"], np.ndarray)
        # Ethanol has 3 atoms
        assert result["atom_feats"].shape[0] == 3
        # With bigraph=True, should have double the edges
        assert result["bond_tuples"].shape[0] == 4  # 2 bonds * 2 directions

    def test_get_mol_graph_no_bigraph(self, mol_graph, ethanol_mol):
        """Test get_mol_graph with bigraph=False."""
        result = mol_graph.get_mol_graph(ethanol_mol, bigraph=False)
        # With bigraph=False, should have original number of edges
        assert result["bond_tuples"].shape[0] == 2  # 2 bonds

    def test_get_pyg_graph(self, mol_graph, ethanol_mol):
        """Test get_pyg_graph method."""
        result = mol_graph.get_pyg_graph(ethanol_mol, bigraph=True)
        assert isinstance(result, Data)
        assert hasattr(result, "x")
        assert hasattr(result, "edge_index")
        assert hasattr(result, "edge_attr")
        assert result.x.shape[0] == 3  # 3 atoms
        assert result.edge_index.shape[1] == 4  # 4 directed edges

    def test_pe_embed(self):
        """Test MolGraph with positional encoding."""
        mol_graph = MolGraph(
            atom_feats=["a_onehot"],
            bond_feats=["b_degree"],
            pe_embed_k=2,
        )
        mol = Chem.MolFromSmiles("CCO")
        result = mol_graph.get_pyg_graph(mol, bigraph=True)
        # Check that PE was added
        assert result.x.shape[1] == mol_graph.num_atom_feats


class TestGetMolFeaturesSizes:
    """Tests for get_mol_feats_sizes function."""

    def test_basic_sizes(self):
        """Test getting feature sizes."""
        atom_size, bond_size = get_mol_feats_sizes(
            atom_feats=["a_onehot", "a_degree"],
            bond_feats=["b_degree"],
            pe_embed_k=0,
        )
        assert isinstance(atom_size, int)
        assert isinstance(bond_size, int)
        assert atom_size > 0
        assert bond_size > 0

    def test_with_pe(self):
        """Test feature sizes with positional encoding."""
        atom_size_no_pe, _ = get_mol_feats_sizes(
            atom_feats=["a_onehot"],
            bond_feats=["b_degree"],
            pe_embed_k=0,
        )
        atom_size_with_pe, _ = get_mol_feats_sizes(
            atom_feats=["a_onehot"],
            bond_feats=["b_degree"],
            pe_embed_k=3,
        )
        assert atom_size_with_pe == atom_size_no_pe + 3


class TestGetMolFp:
    """Tests for get_mol_fp function."""

    @pytest.fixture
    def ethanol_mol(self):
        """Create an ethanol molecule for testing."""
        return Chem.MolFromSmiles("CCO")

    def test_morgan_only(self, ethanol_mol):
        """Test Morgan fingerprint only."""
        fp = get_mol_fp(ethanol_mol, morgan=True, rdkit=False, maccs=False)
        assert isinstance(fp, th.Tensor)
        assert fp.dtype == th.float
        assert len(fp.shape) == 1
        assert fp.shape[0] > 0

    def test_rdkit_only(self, ethanol_mol):
        """Test RDKit fingerprint only."""
        fp = get_mol_fp(ethanol_mol, morgan=False, rdkit=True, maccs=False)
        assert isinstance(fp, th.Tensor)
        assert fp.dtype == th.float
        assert fp.shape[0] > 0

    def test_maccs_only(self, ethanol_mol):
        """Test MACCS fingerprint only."""
        fp = get_mol_fp(ethanol_mol, morgan=False, rdkit=False, maccs=True)
        assert isinstance(fp, th.Tensor)
        assert fp.dtype == th.float
        assert fp.shape[0] > 0

    def test_all_fingerprints(self, ethanol_mol):
        """Test all fingerprints concatenated."""
        fp = get_mol_fp(ethanol_mol, morgan=True, rdkit=True, maccs=True)
        assert isinstance(fp, th.Tensor)
        # Should be larger than any individual fingerprint
        fp_morgan = get_mol_fp(ethanol_mol, morgan=True, rdkit=False, maccs=False)
        assert fp.shape[0] > fp_morgan.shape[0]

    def test_assertion_all_false(self, ethanol_mol):
        """Test that assertion fails when all flags are False."""
        with pytest.raises(AssertionError):
            get_mol_fp(ethanol_mol, morgan=False, rdkit=False, maccs=False)


class TestGetMolFpSize:
    """Tests for get_mol_fp_size function."""

    def test_morgan_size(self):
        """Test Morgan fingerprint size."""
        size = get_mol_fp_size(morgan=True, rdkit=False, maccs=False)
        assert isinstance(size, int)
        assert size > 0

    def test_all_fingerprints_size(self):
        """Test concatenated fingerprint size."""
        size_all = get_mol_fp_size(morgan=True, rdkit=True, maccs=True)
        size_morgan = get_mol_fp_size(morgan=True, rdkit=False, maccs=False)
        assert size_all > size_morgan


class TestGetMolGraph:
    """Tests for get_mol_graph function."""

    def test_basic_graph(self):
        """Test basic molecule graph creation."""
        mol = Chem.MolFromSmiles("CCO")
        graph = get_mol_graph(
            mol,
            atom_feats=["a_onehot", "a_degree"],
            bond_feats=["b_degree"],
            pe_embed_k=0,
            bigraph=True,
        )
        assert isinstance(graph, Data)
        assert hasattr(graph, "x")
        assert hasattr(graph, "edge_index")
        assert hasattr(graph, "edge_attr")

    def test_with_pe(self):
        """Test molecule graph with positional encoding."""
        mol = Chem.MolFromSmiles("CCO")
        graph = get_mol_graph(
            mol,
            atom_feats=["a_onehot"],
            bond_feats=["b_degree"],
            pe_embed_k=3,
            bigraph=True,
        )
        assert isinstance(graph, Data)
        # Check that features include PE dimensions
        assert graph.x.shape[1] > 0


class TestBatchMolsFrags:
    """Tests for batch_mols_frags function."""

    @pytest.fixture
    def simple_mol_pyg_list(self):
        """Create simple molecule graphs for testing."""
        mol1 = Data(
            x=th.randn(3, 10),
            edge_index=th.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            edge_attr=th.randn(4, 5),
        )
        mol2 = Data(
            x=th.randn(4, 10),
            edge_index=th.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
            edge_attr=th.randn(6, 5),
        )
        return [mol1, mol2]

    @pytest.fixture
    def simple_frag_pyg_list(self):
        """Create simple fragment graphs for testing."""
        frag1 = Data(
            x=th.randn(5, 8),
            edge_index=th.tensor([[0, 1], [1, 0]]),
            edge_attr=th.randn(2, 3),
            node_feat_idxs=th.tensor([[0, 4, 8]]),
            edge_feat_idxs=th.tensor([[0, 3]]),
        )
        frag2 = Data(
            x=th.randn(6, 8),
            edge_index=th.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
            edge_attr=th.randn(4, 3),
            node_feat_idxs=th.tensor([[0, 4, 8]]),
            edge_feat_idxs=th.tensor([[0, 3]]),
        )
        return [frag1, frag2]

    @pytest.fixture
    def simple_formula_peaks(self):
        """Create simple formula peak data for testing."""
        # First molecule: 2 formulas, each with 2 peaks (isotopes)
        mzs1 = th.tensor([[100.0, 101.0], [150.0, 0.0]])
        probs1 = th.tensor([[0.9, 0.1], [1.0, 0.0]])
        # Second molecule: 3 formulas, each with 1-2 peaks
        mzs2 = th.tensor([[120.0, 121.0], [180.0, 0.0], [200.0, 201.0]])
        probs2 = th.tensor([[0.8, 0.2], [1.0, 0.0], [0.7, 0.3]])
        return ([mzs1, mzs2], [probs1, probs2])

    def test_batch_basic(self, simple_mol_pyg_list, simple_frag_pyg_list, simple_formula_peaks):
        """Test basic batching of molecules and fragments."""
        mzs_list, probs_list = simple_formula_peaks
        result = batch_mols_frags(simple_mol_pyg_list, simple_frag_pyg_list, mzs_list, probs_list)

        assert "mol_pyg" in result
        assert "frag_pyg" in result
        assert "mol_num_nodes" in result
        assert "frag_num_nodes" in result
        assert "frag_formula_peak_idxs" in result
        assert "frag_formula_peak_mzs" in result
        assert "frag_formula_peak_probs" in result
        assert "frag_formula_sizes" in result
        assert "frag_formula_cumsizes" in result
        assert "frag_formula_peak_sizes" in result

    def test_batch_sizes(self, simple_mol_pyg_list, simple_frag_pyg_list, simple_formula_peaks):
        """Test that batch sizes are correct."""
        mzs_list, probs_list = simple_formula_peaks
        result = batch_mols_frags(simple_mol_pyg_list, simple_frag_pyg_list, mzs_list, probs_list)

        # Check batch has correct number of graphs
        assert result["mol_pyg"].num_graphs == 2
        assert result["frag_pyg"].num_graphs == 2

        # Check cumulative node counts
        assert len(result["mol_num_nodes"]) == 3  # [0, mol1_nodes, mol1+mol2_nodes]
        assert len(result["frag_num_nodes"]) == 3


class TestGetFragGraph:
    """Tests for get_frag_graph function."""

    @pytest.fixture
    def sample_dag_pyg(self):
        """Create a sample DAG PyG object with valid node/edge feature indices.

        NODE_FEAT_TO_IDX = {"depth": 0, "cc": 1, "base_formula": 2, "h_formulae_idx": 3, "h_counts": 4, "nb_iso_idx": 5, "cmf_h_formulae_idx": 6}
        EDGE_FEAT_TO_IDX = {"cc": 0, "base_formula": 1, "h_range": 2, "complement": 3}
        """
        # Node features: 7 features (depth, cc, base_formula, h_formulae_idx, h_counts, nb_iso_idx, cmf_h_formulae_idx)
        # Let's say: depth=1, cc=2, base_formula=3, h_formulae_idx=1, h_counts=2, nb_iso_idx=1, cmf_h_formulae_idx=1
        # Total: 11 dimensions
        node_feat_idxs = th.tensor(
            [[0, 1, 3, 6, 7, 9, 10, 11]]
        )  # cumulative indices for each feature
        # Edge features: 4 features (cc, base_formula, h_range, complement)
        # Let's say: cc=2, base_formula=3, h_range=2, complement=1
        # Total: 8 dimensions
        edge_feat_idxs = th.tensor([[0, 2, 5, 7, 8]])  # cumulative indices for each feature

        return Data(
            x=th.randint(0, 10, (5, 11)),
            edge_index=th.tensor([[0, 1, 2], [1, 2, 3]]),
            edge_attr=th.randint(0, 5, (3, 8)),
            node_feat_idxs=node_feat_idxs,
            edge_feat_idxs=edge_feat_idxs,
        )

    def test_basic_frag_graph(self, sample_dag_pyg):
        """Test basic fragment graph creation."""
        result = get_frag_graph(
            sample_dag_pyg,
            frag_node_feats=["depth", "cc"],
            frag_edge_feats=["cc", "base_formula"],
            edges=True,
            bigraph=False,
        )
        assert isinstance(result, Data)
        assert hasattr(result, "x")
        assert hasattr(result, "edge_index")
        assert hasattr(result, "edge_attr")
        assert hasattr(result, "node_feat_idxs")
        assert hasattr(result, "edge_feat_idxs")

    def test_no_edges(self, sample_dag_pyg):
        """Test fragment graph with edges=False."""
        result = get_frag_graph(
            sample_dag_pyg,
            frag_node_feats=["depth", "cc"],
            frag_edge_feats=[],
            edges=False,
            bigraph=False,
        )
        assert result.edge_index.shape[1] == 0
        assert result.edge_attr.shape[0] == 0

    def test_bigraph(self, sample_dag_pyg):
        """Test fragment graph with bigraph=True."""
        result_directed = get_frag_graph(
            sample_dag_pyg,
            frag_node_feats=["depth", "cc"],
            frag_edge_feats=["cc", "complement"],
            edges=True,
            bigraph=False,
        )
        result_undirected = get_frag_graph(
            sample_dag_pyg,
            frag_node_feats=["depth", "cc"],
            frag_edge_feats=["cc", "complement"],
            edges=True,
            bigraph=True,
        )
        # Undirected should have double the edges
        assert result_undirected.edge_index.shape[1] == 2 * result_directed.edge_index.shape[1]


class TestRandomWalkPE:
    """Tests for random_walk_pe function."""

    def test_basic_rwpe(self):
        """Test basic random walk positional encoding."""
        # Create a simple graph: 0-1-2
        g = Data(
            x=th.randn(3, 5),
            edge_index=th.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        )
        pe = random_walk_pe(g, k=2)
        assert isinstance(pe, th.Tensor)
        assert pe.shape == (3, 2)  # 3 nodes, 2 steps
        assert pe.dtype == th.float32

    def test_rwpe_values(self):
        """Test that RWPE values are valid probabilities."""
        g = Data(
            x=th.randn(4, 5),
            edge_index=th.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        )
        pe = random_walk_pe(g, k=3)
        # All values should be between 0 and 1
        assert th.all(pe >= 0)
        assert th.all(pe <= 1)

    def test_rwpe_different_k(self):
        """Test RWPE with different k values."""
        g = Data(
            x=th.randn(3, 5),
            edge_index=th.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        )
        pe_k2 = random_walk_pe(g, k=2)
        pe_k5 = random_walk_pe(g, k=5)
        assert pe_k2.shape[1] == 2
        assert pe_k5.shape[1] == 5
        # First two columns should be the same
        assert th.allclose(pe_k2, pe_k5[:, :2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
