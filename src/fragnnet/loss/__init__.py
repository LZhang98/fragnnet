"""Functional loss and distance math on sparse mass spectra (was ``model/loss.py``).

Important binning assumption:
    Several distance/similarity losses use binned spectra with a nominal
    support size of ``num_bins = round(mz_max / mz_bin_res)``.
    With the common defaults ``mz_max=1500.0`` and ``mz_bin_res=0.01``,
    this implies a 150,000-bin spectrum grid.

    Pairwise helpers remap to occupied bins to avoid materializing a full
    150,000-wide dense matrix when possible, but the numerical behavior is
    still defined by that underlying bin grid.

The modules import ``loss.padding`` and ``utils`` only, never each other. This initializer
re-exports nothing; import from the defining module.
"""
