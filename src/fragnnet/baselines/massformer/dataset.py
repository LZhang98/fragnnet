"""MassFormer dataset: ``SpecMolDataset`` plus the Graphormer molecule input (``mol_mf``)."""

from __future__ import annotations

import fragnnet.massformer.data_utils as mf_data_utils
from fragnnet.dataset import SpecMolDataset


class SpecMolMFDataset(SpecMolDataset):
    """Spectrum-molecule dataset that also builds MassFormer's Graphormer input.

    With ``mol_params["mf"]`` set, each item carries ``mol_mf`` (``gf_preprocess`` output) and
    collation flattens the batch into ``mol_mf_*`` tensors with MassFormer's ``collator``. The
    shared ``SpecMolDataset`` no longer knows about either step.
    """

    handles_mol_mf = True

    def _process_mol(self, mol_entry) -> dict:
        """Process a molecule entry; add ``mol_mf`` when ``mol_params["mf"]`` is set.

        Args:
            mol_entry: DataFrame row containing molecular data.

        Returns:
            The shared molecule data, plus ``mol_mf`` (a Graphormer ``Data`` object).
        """
        mol_data = super()._process_mol(mol_entry)
        if self.mol_params["mf"]:
            mol_data["mol_mf"] = mf_data_utils.gf_preprocess(mol_entry["mol"], -1)
        return mol_data

    @staticmethod
    def get_collate_fn():
        """Return the collate function for this dataset (dispatch is static, not inherited)."""
        return SpecMolMFDataset.collate_fn

    @staticmethod
    def _special_collate(keys: list[str], collate_data: dict) -> None:
        """Flatten ``mol_mf`` into ``mol_mf_*`` tensors, then apply the shared collation.

        Args:
            keys: Data keys being collated (modified in place).
            collate_data: Lists of values to collate (modified in place).
        """
        if "mol_mf" in keys:
            mol_mf_d = mf_data_utils.collator(collate_data["mol_mf"])
            for k, v in mol_mf_d.items():
                collate_data["mol_mf_" + k] = v
            collate_data.pop("mol_mf")
            keys.remove("mol_mf")
        SpecMolDataset._special_collate(keys, collate_data)

    @staticmethod
    def collate_fn(data_list: list[dict]) -> dict:
        """Collate spectrum-molecule samples, including MassFormer's Graphormer input.

        Args:
            data_list: Data dictionaries from dataset samples.

        Returns:
            Batched data dictionary ready for model input.
        """
        batch_size, keys, collate_data = SpecMolMFDataset._setup_collate(data_list)
        SpecMolMFDataset._special_collate(keys, collate_data)
        SpecMolMFDataset._standard_collate(batch_size, keys, collate_data)
        return collate_data
