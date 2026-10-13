"""Inference and export: load a model from a checkpoint, run it over a dataset, and write the
predictions as pickles, tables or HDF5 (was ``utils/script_utils.py``).

- checkpoint: checkpoint lookup, config overlay, model construction, device.
- export_columns: the export column schema and output-subset options.
- batch_rows: one batch to export rows, and the peak-intensity filter.
- runners: the pkl, tabular, streaming-tabular and HDF5 exporters.

Core: imports no model code; models plug in through ``fragnnet.export_adapter.ExportAdapter``.
"""
