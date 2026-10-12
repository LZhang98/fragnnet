import argparse
import os
import re
from io import BytesIO

import matplotlib.collections as mc
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch as th
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from PIL import Image
from rdkit.Chem import Draw
from tqdm import tqdm

from fragnnet.utils.formula_utils import th_long_to_mask
from fragnnet.utils.mol_utils import get_highlighted_bonds


def font_size_from_ax_height(ax, fraction=6):
    fig = plt.gcf()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax_height = bbox.height
    font_size = int((ax_height / fraction) * 72)
    return font_size


def plot_formula_axes(
    ax,
    formula: str,
    formula_label: str,
    formula_prob=None,
    formula_mz=None,
    font_size=None,
    font="DejaVu Sans Mono",
):
    fig = plt.gcf()

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if font_size == None:
        font_size = font_size_from_ax_height(ax, fraction=6)

    ax_width, ax_height = bbox.width, bbox.height
    # print(ax_width, ax_height)
    ax_width *= fig.dpi
    ax_height *= fig.dpi

    ax.set_xlim(0, ax_width)
    ax.set_ylim(ax_height, 0)

    ax.axis("off")
    font_size_pixel = int(font_size * fig.dpi / 72)
    text_loc = font_size_pixel / 3 * 2
    # ax.text(0, text_loc, f"Peak {peak_id}", size=font_size, horizontalalignment='left', verticalalignment = 'center')

    # circle = matplotlib.patches.Circle((0, 0), radius=1, color='rea')
    # ax.add_patch(circle)

    ax.text(
        10,
        text_loc,
        "$f_{" + formula_label + "}$",
        size=font_size,
        horizontalalignment="left",
        verticalalignment="center",
        bbox=dict(boxstyle="round", ec="black", fc="none", pad=0.15),
    )

    # text_loc += font_size * 4
    # font_size = int((bbox.height / 4 + circle_pad * 2 ) * 72)

    main_text_x = 2.0 * font_size_pixel
    main_text_y_gap = 1.5 * font_size_pixel
    formula_display_str = "$" + formula + "$"
    numbers = re.compile(r"(\d+)")
    formula_display_str = numbers.sub(r"_{\1}", formula_display_str)
    ax.text(
        main_text_x,
        text_loc,
        formula_display_str,
        size=font_size,
        horizontalalignment="left",
        verticalalignment="center",
        fontfamily=font,
    )

    # if formula_prob is not None:
    text_loc += main_text_y_gap
    if formula_mz is not None and pd.notna(formula_mz):
        mz_display_str = f"$m/z={formula_mz:.4f}$"
        ax.text(
            main_text_x,
            text_loc,
            mz_display_str,
            size=font_size,
            horizontalalignment="left",
            verticalalignment="center",
            fontfamily=font,
        )
        text_loc += main_text_y_gap
    if formula_prob is not None:
        formula_prob_display_str = "$P(f_{" + formula_label + "})=" + f"{formula_prob:.3f}" + "$"
        ax.text(
            main_text_x,
            text_loc,
            formula_prob_display_str,
            size=font_size,
            horizontalalignment="left",
            verticalalignment="center",
            fontfamily=font,
        )

    text_loc = 0.85 * ax_height
    formula_node_display_str = "$P(n|f_{" + formula_label + "})=$"
    ax.text(
        main_text_x,
        text_loc,
        formula_node_display_str,
        size=font_size,
        horizontalalignment="left",
        verticalalignment="center",
        fontfamily=font,
    )


def plot_mol_frag_axes(
    ax,
    mol: str,
    atom_highlight_idx: list,
    bond_highlight_idx: list,
    frag_formula_prob: float,
    font_size=None,
    font="DejaVu Sans Mono",
):
    fig = plt.gcf()

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    ax_width, ax_height = bbox.width, bbox.height
    # print(ax_width, ax_height)
    ax_width *= fig.dpi
    ax_height *= fig.dpi

    ax.set_xlim(0, ax_width)
    ax.set_ylim(ax_height, 0)

    image_width = int(ax_width * 0.9)
    image_height = int(ax_height * 0.65)
    image_x_offset = int(0.5 * (ax_width - image_width))
    image_y_offset = 0.0

    # print(image_width, image_height)
    d2d = Draw.MolDraw2DCairo(image_width, image_height)
    options = d2d.drawOptions()
    rgba_color = (0.529, 0.808, 0.98)  # transparent blue
    options.fillHighlights = True
    options.setHighlightColour(rgba_color)
    options.highlightBondWidthMultiplier = 2
    options.clearBackground = False
    d2d.DrawMolecule(mol, highlightAtoms=atom_highlight_idx, highlightBonds=bond_highlight_idx)
    d2d.FinishDrawing()
    bio = BytesIO(d2d.GetDrawingText())
    mol_image = Image.open(bio)

    # (left, right, bottom, top)
    extent = (
        0 + image_x_offset,
        image_width + image_x_offset,
        image_height + image_y_offset,
        image_y_offset,
    )
    ax.imshow(mol_image, extent=extent, interpolation="lanczos")

    if font_size == None:
        font_size = int(bbox.height * 72 / 6)
    text_loc = 0.85 * ax_height
    # formula_node_display_str =  "$P(n|f_{"+ str(peak_id) +"})=$"
    ax.text(
        ax_width / 2,
        text_loc,
        f"{frag_formula_prob:.3f}",
        size=font_size,
        horizontalalignment="center",
        verticalalignment="center",
        fontfamily=font,
    )
    ax.axis("off")


def plot_spectra_axes(
    ax,
    true_mzs,
    true_ints,
    pred_mzs,
    pred_ints,
    bar_width=3,
    font_size=24,
    true_ints_thresh=0,
    pred_ints_thresh=0,
    colors=["green", "red"],
    peak_label_dict=None,
    ann_text=None,
    annotate_min_int=5.0,
):
    mz_max = 2000
    bar_width = 1

    has_true = False
    if true_mzs is not None and len(true_mzs) > 0:
        true_mask = true_ints > true_ints_thresh
        true_mzs = true_mzs[true_mask]
        true_ints = true_ints[true_mask]
        if len(true_mzs) > 0:
            has_true = True

    pred_mask = pred_ints > pred_ints_thresh
    pred_mzs = pred_mzs[pred_mask]
    pred_ints = pred_ints[pred_mask]

    max_mz_vals = []
    if has_true:
        max_mz_vals.append(np.max(true_mzs))
    if len(pred_mzs) > 0:
        max_mz_vals.append(np.max(pred_mzs))

    if len(max_mz_vals) > 0:
        true_mz_max = max(max_mz_vals)
        mz_max = int(np.floor(1.05 * min(mz_max, true_mz_max)))

    if has_true:
        true_mask = true_mzs < mz_max
        true_mzs = true_mzs[true_mask]
        true_ints = true_ints[true_mask]

    pred_mask = pred_mzs < mz_max
    pred_mzs = pred_mzs[pred_mask]
    pred_ints = pred_ints[pred_mask]

    max_int_vals = []
    if has_true:
        max_int_vals.append(np.max(true_ints))
    if len(pred_ints) > 0:
        max_int_vals.append(np.max(pred_ints))

    ints_max = np.max(max_int_vals) if len(max_int_vals) > 0 else 100
    y_max = 1.05 * ints_max

    # Set larger axes ranges
    if has_true:
        ax.set_ylim(-y_max, y_max)
    else:
        ax.set_ylim(0, y_max)
    ax.set_xlim(0, mz_max)

    # Hide y-axis labels beyond ±1000
    # ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: '' if (val > 1000 or val < -1000) else f"{int(val)}"))

    # get peaks
    lines_top, styles_top = [], []

    if has_true:
        mz_mask = np.ones_like(true_mzs, dtype=bool)
        for i in range(len(true_mzs)):
            mz = true_mzs[i]
            ints = true_ints[i]
            line_start = (mz, 0)
            line_end = (mz, ints)
            lines_top.append((line_start, line_end))
            if mz_mask[i]:
                styles_top.append("solid")
            else:
                styles_top.append((0, (1, 1)))

        # print(lines_top[:10])
        lc_top = mc.LineCollection(
            lines_top, colors=colors[0], linewidths=bar_width, linestyles=styles_top
        )
        ax.add_collection(lc_top)

    # add botton peaks and peak ids
    lines_pred = []
    pred_annotations = []

    # peak_label_dict can be a dict {pred_peak_index: label_str}
    if peak_label_dict is None:
        peak_label_dict = {}

    # we maye need change this to bars
    for i in range(len(pred_mzs)):
        mz = pred_mzs[i]
        ints = pred_ints[i]
        line_start = (mz, 0)
        if has_true:
            line_end = (mz, -ints)
        else:
            line_end = (mz, ints)
        lines_pred.append((line_start, line_end))

        # annotate with provided formula label if available
        if i in peak_label_dict and ints >= annotate_min_int:
            label = str(peak_label_dict[i])
            # print(label)
            y_offset = (ints_max * 0.02) if ints > 50 else 50

            if has_true:
                text_y = -ints - y_offset
                va = "top"
            else:
                text_y = ints + y_offset
                va = "bottom"

            pred_annotations.append(
                ax.text(
                    mz,
                    text_y,
                    label,
                    size=font_size,
                    ha="left",
                    va=va,
                    bbox=dict(
                        boxstyle="round",
                        ec="black",
                        fc="none",
                        pad=0.15,
                    ),
                )
            )

    lc_pred = mc.LineCollection(lines_pred, colors=colors[1], linewidths=bar_width)
    ax.add_collection(lc_pred)

    # bars = ax.bar(pred_mzs, -pred_ints, width=bar_width/2, color=colors[1])

    ax.set_axisbelow(True)
    ax.grid(color="black", linewidth=0.1)
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    # ax.tick_params(axis='both', which='both', labelsize='small')
    ax.spines["right"].set_color("none")
    if has_true:
        ax.spines["bottom"].set_position("zero")
    else:
        ax.spines["bottom"].set_position(("data", 0))
    ax.spines["top"].set_color("none")

    # Note: Text overlaps may occur without adjust_text
    leg_font_size = 0.9 * font_size
    leg_kws = {"ncol": 1, "fontsize": leg_font_size, "loc": "upper left"}
    handles = []
    if has_true:
        handles.append(patches.Patch(facecolor=colors[0], label="Ground Truth", alpha=1.0))
    handles.append(patches.Patch(facecolor=colors[1], label="Predicted", alpha=1.0))

    # Overlay NCE/ACE on the spectrum axes if available
    # ann_text = None
    # if nce is not None and pd.notna(nce) and str(nce).strip():
    #   ann_text = f'NCE: {nce}'
    # elif ace is not None and pd.notna(ace) and str(ace).strip():
    #   ann_text = f'ACE: {ace}'
    if ann_text is not None:
        ax.text(
            0.99,
            0.02,
            ann_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=16,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    ax.legend(handles=handles, **leg_kws)
    # Remove 0 from x-axis ticks
    xticks = ax.get_xticks()
    ax.set_xticks(xticks[1:-1])

    # Show absolute values on y-axis (mirror plot)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{abs(y):.0f}"))


def load_dataframe(file_path_gz, file_path_pkl, description):
    """Load dataframe from .pkl.gz or .pkl file"""
    if os.path.exists(file_path_gz):
        df = pd.read_pickle(file_path_gz)
        print(f"Loaded {description} with {len(df)} entries from {file_path_gz}")
    elif os.path.exists(file_path_pkl):
        df = pd.read_pickle(file_path_pkl)
        print(f"Loaded {description} with {len(df)} entries from {file_path_pkl}")
    else:
        raise FileNotFoundError(
            f"Could not find {description} file. Looked for both {file_path_gz} and {file_path_pkl}"
        )
    return df


def load_and_prepare_data(args):
    """Load and merge all reference data"""
    proc_dp = args.proc_dp

    # Load mol_df
    mol_df_path_gz = os.path.join(proc_dp, "mol_df.pkl.gz")
    mol_df_path_pkl = os.path.join(proc_dp, "mol_df.pkl")
    mol_df = load_dataframe(mol_df_path_gz, mol_df_path_pkl, "mol_df")

    # Load spec_df
    spec_df_path_gz = os.path.join(proc_dp, "spec_df.pkl.gz")
    spec_df_path_pkl = os.path.join(proc_dp, "spec_df.pkl")
    spec_df = load_dataframe(spec_df_path_gz, spec_df_path_pkl, "spec_df")

    # Merge with mol_df
    has_input_mol_id = "input_mol_id" in mol_df.columns
    if has_input_mol_id:
        print("mol_df contains 'input_mol_id' column")
        spec_df = spec_df.merge(
            mol_df[["mol_id", "smiles", "inchikey_s", "input_mol_id", "mol"]],
            on="mol_id",
            how="left",
        )
    else:
        spec_df = spec_df.merge(
            mol_df[["mol_id", "smiles", "inchikey_s", "mol"]], on="mol_id", how="left"
        )

    # Load pred_df
    pred_df = pd.read_pickle(args.predicted_fp)
    print(f"Loaded pred_df with {len(pred_df)} predictions")
    if "mol_id" not in pred_df.columns:
        pred_df = pred_df.merge(spec_df, on="spec_id", how="left")
    else:
        pred_df = pred_df.merge(spec_df, on=["spec_id", "mol_id"], how="left")
    print(f"Merged pred_df now has {len(pred_df)} rows")

    # Add pred_peaks column
    def get_pred_peaks(x):
        mzs = x.get("pred_mzs")
        ints = x.get("pred_ints")
        if isinstance(mzs, (float, np.floating)) and np.isnan(mzs):
            return np.array([])
        if isinstance(ints, (float, np.floating)) and np.isnan(ints):
            return np.array([])
        return np.stack([np.array(mzs), np.array(ints)], axis=1)

    pred_df.loc[:, "pred_peaks"] = pred_df.apply(get_pred_peaks, axis=1)

    return pred_df


def parse_peaks_from_string(peaks_data):
    """Parse true peaks from various data formats"""
    # Check if data is NaN/None (handle both scalar and array cases)
    try:
        is_true_peak_na = peaks_data is None or (
            isinstance(peaks_data, (float, np.floating)) and np.isnan(peaks_data)
        )
    except:
        is_true_peak_na = False

    if is_true_peak_na:
        # No true peaks available
        return np.array([]), np.array([])
    elif isinstance(peaks_data, str):
        # String format: "mz1 int1\nmz2 int2\n..."
        if peaks_data.strip():
            peak_lines = peaks_data.strip().split("\n")
            true_mzs, true_ints = [], []
            for line in peak_lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            true_mzs.append(mz)
                            true_ints.append(intensity)
                        except ValueError:
                            continue
            return np.array(true_mzs), np.array(true_ints)
        else:
            return np.array([]), np.array([])
    elif isinstance(peaks_data, (list, np.ndarray)):
        # Array format: assume it's already parsed peaks data
        try:
            if len(peaks_data) > 0 and isinstance(peaks_data[0], (list, tuple)):
                # Format: [(mz1, int1), (mz2, int2), ...]
                true_mzs = [peak[0] for peak in peaks_data]
                true_ints = [peak[1] for peak in peaks_data]
                return np.array(true_mzs), np.array(true_ints)
            else:
                # Unknown format
                return np.array([]), np.array([])
        except:
            return np.array([]), np.array([])
    else:
        # Unknown data type
        return np.array([]), np.array([])


def process_prediction_data(pred_data):
    """Process prediction data to extract formulas, nodes, and probabilities"""

    # Check if required data is available (not NaN)
    required_keys = [
        "pred_formula_formula_idxs",
        "pred_formula_strs",
        "frag_formula_peak_mzs",
        "frag_formula_peak_idxs",
        "pred_formula_probs",
        "pred_joint_formula_idxs",
        "pred_joint_node_idxs",
        "pred_formula_node_probs",
        "pred_node_formula_probs",
        "pred_node_ccs",
        "pred_node_node_idxs",
        "pred_node_probs",
    ]

    for key in required_keys:
        val = pred_data.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return {}, {}, {}, {}

    # Get formula dict
    # TODO double check isotope handling
    formula_dict = {
        i: {"formula": s, "mz": m, "peak_idx": pi, "prob": p}
        for i, s, m, pi, p in zip(
            pred_data["pred_formula_formula_idxs"],
            pred_data["pred_formula_strs"],
            pred_data["frag_formula_peak_mzs"],  # not sure what will heappen if isotope is enabled
            pred_data["frag_formula_peak_idxs"],  # not sure what will heappen if isotope is enabled
            pred_data["pred_formula_probs"],
        )
    }

    # Sort by prob
    formula_dict = dict(
        sorted(formula_dict.items(), key=lambda item: item[1]["prob"], reverse=True)
    )

    # Map formula to nodes
    formula_node_map = {i: [] for i in set(pred_data["pred_joint_formula_idxs"])}
    # for f_id, n_id in zip(pred_data['pred_joint_formula_idxs'], pred_data['pred_joint_node_idxs']):
    #   formula_node_map[f_id].append(n_id)

    pred_formula_node_probs = pred_data["pred_formula_node_probs"]
    pred_node_formula_probs = pred_data["pred_node_formula_probs"]
    pred_joint_formula_idxs = pred_data["pred_joint_formula_idxs"]
    pred_joint_node_idxs = pred_data["pred_joint_node_idxs"]

    node_formula_prob_dict = {}
    formula_node_prob_dict = {}
    for f_id, n_id, f_n_prob, n_f_prob in zip(
        pred_joint_formula_idxs,
        pred_joint_node_idxs,
        pred_formula_node_probs,
        pred_node_formula_probs,
    ):
        node_formula_prob_dict[(f_id, n_id)] = n_f_prob
        formula_node_prob_dict[(f_id, n_id)] = f_n_prob
        formula_node_map[f_id].append(n_id)

    # Get node masks
    frag_node_masks = th_long_to_mask(pred_data["pred_node_ccs"])
    all_atom_ids = th.arange(0, frag_node_masks.shape[1]).to(th.long)
    node_dict = {
        i: {"pred_node_probs": p, "pred_node_ccs": all_atom_ids[c].tolist()}
        for i, p, c in zip(
            pred_data["pred_node_node_idxs"], pred_data["pred_node_probs"], frag_node_masks
        )
    }

    return formula_dict, formula_node_map, node_dict, formula_node_prob_dict


def build_display_data(formula_dict, formula_node_map, node_dict, formula_node_prob_dict, mol):
    """Build display data structure for plotting"""
    display_data = []

    for _, f_id in enumerate(formula_dict.keys()):
        formula_data = {
            "formula_idx": f_id,
            "formula": formula_dict[f_id]["formula"],
            "formula_prob": formula_dict[f_id]["prob"],
            "formula_mass": formula_dict[f_id]["mz"],
            "peak_idx": formula_dict[f_id]["peak_idx"],
            "frag": [],
        }

        for n_id in formula_node_map[f_id]:
            if n_id not in node_dict:
                continue
            pred_node_prob = node_dict[n_id]["pred_node_probs"]
            frag_atom_ids = node_dict[n_id]["pred_node_ccs"]
            formula_data["frag"].append(
                {
                    "node_atom_indices": frag_atom_ids,
                    "node_bond_indices": get_highlighted_bonds(mol, frag_atom_ids),
                    "node_prob": pred_node_prob,
                    "formula_node_prob": formula_node_prob_dict.get((f_id, n_id), 0.0),
                }
            )
        # sort fragments by formula_node_prob
        formula_data["frag"] = sorted(
            formula_data["frag"], key=lambda x: x["formula_node_prob"], reverse=True
        )
        display_data.append(formula_data)

    return display_data


def plot_mirror_spectra(
    pred_data,
    display_data,
    output_dir,
    figsize=(24, 16),
    dpi=300,
    spectra_plot_width=12,
    n_formula_frag_cols=3 + 1,  # formula + 3 fragments
    n_formula_frag_rows=8,
):
    """Generate and save mirror plot for a single spectrum"""

    # check figure size
    assert spectra_plot_width <= figsize[0], (
        "spectra_plot_width must be less than or equal to the width of the figure"
    )
    smiles = pred_data["smiles"]
    mol = pred_data["mol"]
    spec_id = pred_data["spec_id"]
    mol_id = pred_data["mol_id"]

    # Extract predicted spectrum data
    pred_mzs_raw = pred_data.get("pred_mzs")
    pred_ints_raw = pred_data.get("pred_ints")

    if isinstance(pred_mzs_raw, (float, np.floating)) and np.isnan(pred_mzs_raw):
        pred_spec_mzs = np.array([])
    else:
        pred_spec_mzs = np.array(pred_mzs_raw)

    if isinstance(pred_ints_raw, (float, np.floating)) and np.isnan(pred_ints_raw):
        pred_spec_ints = np.array([])
    else:
        pred_spec_ints = np.array(pred_ints_raw)

    # Parse true peaks
    peaks_data = pred_data.get("peaks", None)
    true_spec_mzs, true_spec_ints = parse_peaks_from_string(peaks_data)

    # Skip if no data available
    if len(pred_spec_mzs) == 0 and len(true_spec_mzs) == 0:
        return False

    # Setup plot layout

    # Build title with additional metadata
    title_parts = [f"Spec ID: {spec_id}", f"Mol ID: {mol_id}"]

    # Add collision energy if available
    col_energy = pred_data.get("col_energy", None)
    if col_energy is not None and pd.notna(col_energy) and str(col_energy).strip():
        title_parts.append(f"CE: {col_energy}")

    # Prefer normalized CE (NCE) or absolute CE (ACE) if present
    nce = pred_data.get("nce", None)
    ace = pred_data.get("ace", None)
    if nce is not None and pd.notna(nce) and str(nce).strip():
        title_parts.append(f"NCE: {nce}")
    elif ace is not None and pd.notna(ace) and str(ace).strip():
        title_parts.append(f"ACE: {ace}")

    # Add adduct/precursor type if available
    prec_type = pred_data.get("prec_type", None)
    if prec_type is not None and pd.notna(prec_type) and str(prec_type).strip():
        title_parts.append(f"Adduct: {prec_type}")

    # Add instrument type if available
    inst_type = pred_data.get("inst_type", None)
    if inst_type is not None and pd.notna(inst_type) and str(inst_type).strip():
        title_parts.append(f"Inst: {inst_type}")

    title = " | ".join(title_parts)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.suptitle(title, fontsize=28, fontweight="bold")
    grand_gs = GridSpec(ncols=figsize[0], nrows=1, figure=fig)
    left_gs = grand_gs[0:spectra_plot_width].subgridspec(ncols=1, nrows=1)
    right_gs = grand_gs[spectra_plot_width:].subgridspec(
        ncols=n_formula_frag_cols, nrows=n_formula_frag_rows
    )

    # Build peak label map from display_data: {peak_idx: formula}
    peak_label_dict = {}
    for fd in display_data:
        p_idx = fd.get("peak_idx", None)
        f_idx = fd.get("formula_idx", None)
        if p_idx is not None and f_idx is not None and str(f_idx).strip():
            peak_label_dict[int(p_idx)] = r"$f_{" + str(f_idx) + r"}$"

    # Plot formulas and fragments
    auto_font_size = 16
    for f_idx, formula_data in enumerate(display_data):
        if f_idx >= n_formula_frag_rows:
            break
        formula = formula_data["formula"]
        # null formula check (this should not happen)
        if formula is None or formula == "":
            continue

        ax = fig.add_subplot(right_gs[f_idx, 0])
        auto_font_size = (
            font_size_from_ax_height(ax, fraction=6) if auto_font_size is None else auto_font_size
        )
        # print(f"Plotting formula {f_idx}: {formula} for peak {peak_id}")
        plot_formula_axes(
            ax,
            formula,
            str(formula_data["formula_idx"]),
            formula_data["formula_prob"],
            formula_data.get("formula_mass"),
            font_size=auto_font_size,
        )

        for frag_idx, frag in enumerate(formula_data["frag"]):
            if frag_idx + 1 >= n_formula_frag_cols:
                break
            ax = fig.add_subplot(right_gs[f_idx, frag_idx + 1])
            h_atom_list = frag["node_atom_indices"]
            h_bond_list = frag["node_bond_indices"]
            plot_mol_frag_axes(
                ax,
                mol,
                h_atom_list,
                h_bond_list,
                frag["formula_node_prob"],
                font_size=auto_font_size,
            )

    # Plot spectrum
    ax = fig.add_subplot(left_gs[0], facecolor="white")
    plot_spectra_axes(
        ax,
        true_spec_mzs,
        true_spec_ints,
        pred_spec_mzs,
        pred_spec_ints,
        bar_width=1,
        font_size=auto_font_size,
        true_ints_thresh=0,
        pred_ints_thresh=0,
        colors=["blue", "red"],
        peak_label_dict=peak_label_dict,
        ann_text=None,
    )

    # Save plot
    plot_filename = f"mirror_plot_spec_{spec_id}_mol_{mol_id}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved mirror plot: {plot_path}")
    # print(f"  True peaks: {len(true_spec_mzs)}, Predicted peaks: {len(pred_spec_mzs)}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predicted_fp",
        type=str,
        required=False,
        help="Predicted data pickle file path",
        default="output/inference_eval/32n55f2m/predictions_test.pkl",
    )
    parser.add_argument(
        "--proc_dp",
        type=str,
        required=False,
        help="Processed data directory",
        default="data/proc/nist20",
    )
    parser.add_argument(
        "--save_dp",
        type=str,
        required=False,
        help="Directory to save plots",
        default="output/spectra_plots_test",
    )
    parser.add_argument("--has_truth", type=str, required=False, help="True data pickle file path")
    parser.add_argument(
        "--num_plots", type=int, required=False, help="Number of plots to generate", default=20
    )

    args = parser.parse_args()

    # Load and prepare data
    pred_df = load_and_prepare_data(args)

    output_dir = args.save_dp
    os.makedirs(output_dir, exist_ok=True)

    # Limit to first few examples for testing
    num_plots = min(args.num_plots, len(pred_df))

    for idx, pred_data in tqdm(pred_df.head(num_plots).iterrows(), total=num_plots):
        print(pred_data)
        # exit()
        # Process prediction data
        formula_dict, formula_node_map, node_dict, formula_node_prob_dict = process_prediction_data(
            pred_data
        )

        # Build display data
        display_data = build_display_data(
            formula_dict, formula_node_map, node_dict, formula_node_prob_dict, pred_data["mol"]
        )

        # Generate and save plot
        plot_mirror_spectra(pred_data, display_data, output_dir)

    print(f"\nCompleted generating {num_plots} mirror plots in {output_dir}")
