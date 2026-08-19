"""Tests for molecule featurizer (monroe/model/featurizer.py)."""

import numpy as np
import pytest
from rdkit import Chem

from monroe.model.constants import (
    BOND_TYPES,
    EDGE_FEAT_LIST_ONE_HOT,
    NODE_FEAT_LIST_FLOAT,
    NODE_FEAT_LIST_ONE_HOT,
)
from monroe.model.featurizer import (
    _align_conformers,
    build_from_mol_and_coords,
    build_single_graph,
    get_edge_features_codes,
    get_node_features_codes,
    get_stereo_virtual_edges,
    symmetrize_edges,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WATER_INCHI = "InChI=1S/H2O/h1H2"
METHANE_INCHI = "InChI=1S/CH4/h1H4"
ETHANOL_INCHI = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
BENZENE_INCHI = "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H"


def _mol_with_3d(inchi: str) -> Chem.Mol:
    """Build an RDKit mol with Hs and a 3D conformer."""
    mol = Chem.MolFromInchi(inchi, sanitize=False)
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    from rdkit.Chem import AllChem
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol

# ---------------------------------------------------------------------------
# get_node_features_codes
# ---------------------------------------------------------------------------

class TestNodeFeatures:
    def test_water_shapes(self):
        mol = _mol_with_3d(WATER_INCHI)
        feat_float, feat_codes = get_node_features_codes(mol)

        n_atoms = mol.GetNumAtoms()  # 3 (O + 2H)
        assert feat_float.shape == (n_atoms, len(NODE_FEAT_LIST_FLOAT))
        assert feat_codes.shape == (n_atoms, len(NODE_FEAT_LIST_ONE_HOT))
        assert feat_float.dtype == np.float32
        assert feat_codes.dtype == np.uint8

    def test_no_nans(self):
        mol = _mol_with_3d(ETHANOL_INCHI)
        feat_float, _ = get_node_features_codes(mol)
        assert not np.isnan(feat_float).any()

    def test_methane_atom_count(self):
        mol = _mol_with_3d(METHANE_INCHI)
        feat_float, _ = get_node_features_codes(mol)
        assert feat_float.shape[0] == 5  # C + 4H

    def test_benzene_aromaticity(self):
        """All 6 carbons in benzene should be aromatic."""
        mol = _mol_with_3d(BENZENE_INCHI)
        _, feat_codes = get_node_features_codes(mol)

        # is_aromatic is a categorical feature
        arom_idx = list(NODE_FEAT_LIST_ONE_HOT.keys()).index("is_aromatic")
        carbon_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
        for ci in carbon_indices:
            # code 0 means the value matched the first class (which is 1 = aromatic)
            assert feat_codes[ci, arom_idx] == 0, f"Carbon {ci} not marked aromatic"

    def test_benzene_ring_membership(self):
        """All 6 carbons in benzene should be in a ring."""
        mol = _mol_with_3d(BENZENE_INCHI)
        _, feat_codes = get_node_features_codes(mol)

        ring_idx = list(NODE_FEAT_LIST_ONE_HOT.keys()).index("is_in_ring")
        carbon_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
        for ci in carbon_indices:
            assert feat_codes[ci, ring_idx] == 0  # code 0 = matched [1] = True


# ---------------------------------------------------------------------------
# get_edge_features_codes
# ---------------------------------------------------------------------------

class TestEdgeFeatures:
    def test_shapes_match_bonds(self):
        mol = _mol_with_3d(ETHANOL_INCHI)
        edge_codes, edge_index = get_edge_features_codes(mol)

        n_bonds = mol.GetNumBonds()
        assert edge_codes.shape == (n_bonds, len(EDGE_FEAT_LIST_ONE_HOT))
        assert edge_index.shape == (2, n_bonds)
        assert edge_codes.dtype == np.uint8
        assert edge_index.dtype == np.int32

    def test_edge_indices_in_range(self):
        mol = _mol_with_3d(BENZENE_INCHI)
        _, edge_index = get_edge_features_codes(mol)

        n_atoms = mol.GetNumAtoms()
        assert edge_index.min() >= 0
        assert edge_index.max() < n_atoms

    def test_water_has_two_bonds(self):
        mol = _mol_with_3d(WATER_INCHI)
        edge_codes, _ = get_edge_features_codes(mol)
        assert edge_codes.shape[0] == 2  # O-H, O-H

    def test_benzene_bond_types(self):
        """Benzene should have aromatic bonds between carbons."""
        mol = _mol_with_3d(BENZENE_INCHI)
        edge_codes, edge_index = get_edge_features_codes(mol)

        bond_type_idx = list(EDGE_FEAT_LIST_ONE_HOT.keys()).index("bond_type")
        aromatic_code = BOND_TYPES.index(Chem.rdchem.BondType.AROMATIC)

        # Find edges between two carbon atoms
        carbon_set = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6}
        for bi in range(edge_codes.shape[0]):
            src, dst = edge_index[0, bi], edge_index[1, bi]
            if src in carbon_set and dst in carbon_set:
                assert edge_codes[bi, bond_type_idx] == aromatic_code


# ---------------------------------------------------------------------------
# symmetrize_edges
# ---------------------------------------------------------------------------

class TestSymmetrizeEdges:
    def test_doubles_non_stereo_edges(self):
        """Non-stereo edges should be doubled (forward + reverse)."""
        mol = _mol_with_3d(WATER_INCHI)
        edge_codes, edge_index = get_edge_features_codes(mol)

        sym_ei, sym_ec = symmetrize_edges(edge_index, edge_codes)

        # Water has 2 bonds, no stereo → should become 4 directed edges
        assert sym_ei.shape[1] == 4
        assert sym_ec.shape[0] == 4

    def test_reverse_edges_present(self):
        mol = _mol_with_3d(WATER_INCHI)
        edge_codes, edge_index = get_edge_features_codes(mol)
        sym_ei, _ = symmetrize_edges(edge_index, edge_codes)

        edges = set(zip(sym_ei[0], sym_ei[1]))
        # Every (u,v) should have a (v,u)
        for u, v in list(edges):
            assert (v, u) in edges

    def test_stereo_edges_not_doubled(self):
        """Edges with bond_type == stereo_code should not be duplicated."""
        # Create a fake edge array with one stereo edge
        bond_type_idx = list(EDGE_FEAT_LIST_ONE_HOT.keys()).index("bond_type")
        stereo_code = np.uint8(len(BOND_TYPES))

        edge_index = np.array([[0, 1], [1, 2]], dtype=np.int32).T  # 2 edges
        edge_codes = np.zeros((2, len(EDGE_FEAT_LIST_ONE_HOT)), dtype=np.uint8)
        edge_codes[0, bond_type_idx] = 0  # normal bond
        edge_codes[1, bond_type_idx] = stereo_code  # stereo bond

        sym_ei, sym_ec = symmetrize_edges(edge_index, edge_codes)

        # Original 2 edges + 1 reverse (only for the normal bond) = 3
        assert sym_ei.shape[1] == 3


# ---------------------------------------------------------------------------
# get_stereo_virtual_edges
# ---------------------------------------------------------------------------

class TestStereoVirtualEdges:
    def test_water_has_no_stereo(self):
        mol = _mol_with_3d(WATER_INCHI)
        ei_extra, ec_extra = get_stereo_virtual_edges(mol)
        assert ei_extra.shape[1] == 0
        assert ec_extra.shape[0] == 0

    def test_returns_correct_shapes(self):
        mol = _mol_with_3d(ETHANOL_INCHI)
        ei_extra, ec_extra = get_stereo_virtual_edges(mol)
        assert ei_extra.shape[0] == 2  # (2, E_s)
        assert ec_extra.shape[1] == len(EDGE_FEAT_LIST_ONE_HOT)
        assert ei_extra.shape[1] == ec_extra.shape[0]  # consistent count


# ---------------------------------------------------------------------------
# _align_conformers (numpy Kabsch)
# ---------------------------------------------------------------------------

class TestAlignConformers:
    def test_identity(self):
        ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        aligned = _align_conformers(ref, ref.copy())
        np.testing.assert_allclose(aligned, ref, atol=1e-10)

    def test_recover_translation(self):
        ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        moving = ref + np.array([5.0, -3.0, 7.0])
        aligned = _align_conformers(ref, moving)
        np.testing.assert_allclose(aligned, ref, atol=1e-10)

    def test_recover_rotation(self):
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((10, 3))
        # 90-degree rotation around Z
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        moving = ref @ R.T
        aligned = _align_conformers(ref, moving)
        np.testing.assert_allclose(aligned, ref, atol=1e-10)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            _align_conformers(np.zeros((3, 3)), np.zeros((4, 3)))

    def test_preserves_dtype(self):
        ref = np.zeros((3, 3), dtype=np.float32)
        aligned = _align_conformers(ref, ref.copy())
        assert aligned.dtype == np.float32


# ---------------------------------------------------------------------------
# build_from_mol_and_coords
# ---------------------------------------------------------------------------

class TestBuildFromMolAndCoords:
    def test_dict_keys(self):
        mol = _mol_with_3d(WATER_INCHI)
        n_atoms = mol.GetNumAtoms()
        pos_rdkit = np.zeros((n_atoms, 3), dtype=np.float32)

        result = build_from_mol_and_coords(mol, pos_rdkit, stereo_augmentation=False)

        expected_keys = {"node_float", "node_codes", "pos_rdkit", "pos", "edge_index", "edge_codes"}
        assert set(result.keys()) == expected_keys

    def test_pos_none_when_omitted(self):
        mol = _mol_with_3d(METHANE_INCHI)
        n_atoms = mol.GetNumAtoms()
        pos_rdkit = np.zeros((n_atoms, 3), dtype=np.float32)

        result = build_from_mol_and_coords(mol, pos_rdkit, pos=None)
        assert result["pos"] is None

    def test_pos_present_when_provided(self):
        mol = _mol_with_3d(METHANE_INCHI)
        n_atoms = mol.GetNumAtoms()
        pos = np.ones((n_atoms, 3), dtype=np.float32)
        pos_rdkit = np.zeros((n_atoms, 3), dtype=np.float32)

        result = build_from_mol_and_coords(mol, pos_rdkit, pos=pos)
        np.testing.assert_array_equal(result["pos"], pos)

    def test_symmetrize_flag(self):
        mol = _mol_with_3d(WATER_INCHI)
        n_atoms = mol.GetNumAtoms()
        pos_rdkit = np.zeros((n_atoms, 3), dtype=np.float32)

        unsym = build_from_mol_and_coords(mol, pos_rdkit, stereo_augmentation=False, symmetrize=False)
        sym = build_from_mol_and_coords(mol, pos_rdkit, stereo_augmentation=False, symmetrize=True)

        # Symmetrized should have more edges
        assert sym["edge_index"].shape[1] > unsym["edge_index"].shape[1]


# ---------------------------------------------------------------------------
# build_single_graph (end-to-end, inference mode)
# ---------------------------------------------------------------------------

class TestBuildSingleGraph:
    """End-to-end tests for inference-mode graph construction."""

    @pytest.mark.slow
    def test_water(self):
        g = build_single_graph(WATER_INCHI)

        assert g["node_float"].shape[0] == 3  # O + 2H
        assert g["node_float"].shape[1] == len(NODE_FEAT_LIST_FLOAT)
        assert g["node_codes"].shape[1] == len(NODE_FEAT_LIST_ONE_HOT)
        assert g["pos_rdkit"].shape == (3, 3)
        assert g["pos"] is None  # inference mode
        assert g["edge_index"].shape[0] == 2
        assert g["edge_codes"].shape[1] == len(EDGE_FEAT_LIST_ONE_HOT)

    @pytest.mark.slow
    def test_methane(self):
        g = build_single_graph(METHANE_INCHI)
        assert g["node_float"].shape[0] == 5  # C + 4H

    @pytest.mark.slow
    def test_no_nans_in_features(self):
        g = build_single_graph(ETHANOL_INCHI)
        assert not np.isnan(g["node_float"]).any()
        assert not np.isnan(g["pos_rdkit"]).any()

    @pytest.mark.slow
    def test_invalid_inchi_raises(self):
        with pytest.raises((ValueError, Exception)):
            build_single_graph("InChI=1S/INVALID")

    @pytest.mark.slow
    def test_dtypes(self):
        g = build_single_graph(WATER_INCHI)
        assert g["node_float"].dtype == np.float32
        assert g["node_codes"].dtype == np.uint8
        assert g["pos_rdkit"].dtype == np.float32
        assert g["edge_index"].dtype == np.int32
        assert g["edge_codes"].dtype == np.uint8
