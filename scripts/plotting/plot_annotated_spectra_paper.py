import argparse
import pickle
import re
from io import BytesIO
from math import ceil

import matplotlib.collections as mc
import matplotlib.patches as patches
import numpy as np
import torch as th
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

from fragnnet.utils.frag_utils import load_frag_d, th_long_to_mask
from fragnnet.utils.plot_utils import get_highlighted_bonds, get_mol_im
from fragnnet.utils.spec_utils import *


def plot_formula_axes(
    ax,
    formula: str,
    peak_id: str | int,
    formula_prob=None,
    font_size=None,
    font="DejaVu Sans Mono",
    circle_pad=0.1,
):
    fig = plt.gcf()

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax_width, ax_height = bbox.width, bbox.height
    # print(ax_width, ax_height)
    ax_width *= fig.dpi
    ax_height *= fig.dpi

    ax.set_xlim(0, ax_width)
    ax.set_ylim(ax_height, 0)

    ax.axis("off")
    if font_size == None:
        font_size = int(bbox.height * 72 / 4)
    font_size_pixel = int(font_size * fig.dpi / 72)
    text_loc = font_size_pixel / 3 * 2
    # ax.text(0, text_loc, f"Peak {peak_id}", size=font_size, horizontalalignment='left', verticalalignment = 'center')

    # circle = matplotlib.patches.Circle((0, 0), radius=1, color='rea')
    # ax.add_patch(circle)

    ax.text(
        10,
        text_loc,
        r"$f_{" + str(peak_id) + "}$",
        size=font_size,
        horizontalalignment="left",
        verticalalignment="center",
        bbox=dict(
            boxstyle="circle",
            ec=(1.0, 0.5, 0.5),
            fc=(1.0, 0.8, 0.8),
            pad=circle_pad,
        ),
    )

    # text_loc += font_size * 4
    # font_size = int((bbox.height / 4 + circle_pad * 2 ) * 72)

    formula_display_str = "$" + formula + "$"
    numbers = re.compile(r"(\d+)")
    formula_display_str = numbers.sub(r"_{\1}", formula_display_str)
    ax.text(
        1.25 * font_size_pixel,
        text_loc,
        formula_display_str,
        size=font_size,
        horizontalalignment="left",
        verticalalignment="center",
        fontfamily=font,
    )

    # if formula_prob is not None:
    text_loc += 1.25 * font_size_pixel
    formula_prob_display_str = "$P(f_{" + str(peak_id) + "})=" + f"{formula_prob:.3f}" + "$"
    ax.text(
        1.25 * font_size_pixel,
        text_loc,
        formula_prob_display_str,
        size=font_size,
        horizontalalignment="left",
        verticalalignment="center",
        fontfamily=font,
    )

    text_loc = 0.85 * ax_height
    formula_node_display_str = "$P(n|f_{" + str(peak_id) + "})=$"
    ax.text(
        1.25 * font_size_pixel,
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
    options.highlightBondWidthMultiplier = 20
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
        font_size = int(bbox.height * 72 / 4)

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
    bar_width=1,
    font_size=24,
    true_ints_thresh=0,
    pred_ints_thresh=0,
    colors=["green", "red"],
    circle_pad=0.1,
    peak_labels=None,
):
    mz_max = 1000
    bar_width = 1
    # true_ints_thresh = 0
    # pred_ints_thresh = 0

    true_mask = true_ints > true_ints_thresh
    true_mzs = true_mzs[true_mask]
    true_ints = true_ints[true_mask]
    pred_mask = pred_ints > pred_ints_thresh
    pred_mzs = pred_mzs[pred_mask]
    pred_ints = pred_ints[pred_mask]
    true_mz_max = max(np.max(true_mzs), np.max(pred_mzs))

    mz_max = int(np.floor(1.05 * min(mz_max, true_mz_max)))
    true_mask = true_mzs < mz_max
    pred_mask = pred_mzs < mz_max
    true_mzs = true_mzs[true_mask]
    true_ints = true_ints[true_mask]
    pred_mzs = pred_mzs[pred_mask]
    pred_ints = pred_ints[pred_mask]

    x_max = mz_max
    ints_max = np.max(true_ints)
    y_max = 1.05 * ints_max

    #
    ax.set_ylim(-y_max, y_max)
    ax.set_xlim(0, x_max)

    # get peaks
    lines_top, styles_top = [], []

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
    lines_bottom = []
    bottom_annotations = []
    bottom_annotations_x = []
    bottom_annotations_y = []

    if peak_labels is None:
        peak_labels = []

    # we maye need change this to bars
    for i in range(len(pred_mzs)):
        mz = pred_mzs[i]
        ints = pred_ints[i]
        line_start = (mz, 0)
        line_end = (mz, -ints)
        lines_bottom.append((line_start, line_end))

        if i in peak_labels:
            bottom_annotations.append(
                ax.text(
                    mz,
                    -ints - ints_max * 0.02,
                    "$f_{" + str(peak_labels[i]) + "}$",
                    size=font_size,
                    ha="left",
                    va="top",
                    bbox=dict(
                        boxstyle="circle", ec=(1.0, 0.5, 0.5), fc=(1.0, 0.8, 0.8), pad=circle_pad
                    ),
                )
            )
            bottom_annotations_x.append(mz)
            bottom_annotations_y.append(-ints)

    lc_bottom = mc.LineCollection(lines_bottom, colors=colors[1], linewidths=bar_width)
    lc_bottom = ax.add_collection(lc_bottom)

    # bars = ax.bar(pred_mzs, -pred_ints, width=bar_width/2, color=colors[1])

    ax.set_axisbelow(True)
    ax.grid(color="black", linewidth=0.1)
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    # ax.tick_params(axis='both', which='both', labelsize='small')
    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_position("zero")
    ax.spines["top"].set_color("none")

    # Note: Text overlaps may occur without adjust_text

    leg_font_size = 0.9 * font_size
    leg_kws = {"ncol": 1, "fontsize": leg_font_size, "loc": "upper left"}
    handles = [
        patches.Patch(facecolor=colors[0], label="Ground Truth", alpha=1.0),
        patches.Patch(facecolor=colors[1], label="Predicted", alpha=1.0),
    ]
    ax.legend(handles=handles, **leg_kws)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predicted_fp",
        type=str,
        required=False,
        help="Predicted data pickle file path",
        default="data/predicted/nist20_d4_wr00_inchikey/kme8fry7_fraggnn_d4_nodemlp_mnce_nocp.pkl.gz",
    )
    parser.add_argument(
        "--frag_dp",
        type=str,
        required=False,
        help="Fragment data directory path",
        default="data/frag/nist20_d4_h4_isoFalse/dags",
    )

    args = parser.parse_args()
    all_pred_data = []
    with open(args.predicted_fp, "rb") as file_in:
        all_pred_data = pickle.load(file_in)
    # spec_unique_ids = [10731,12205,12946,12955]
    # mol_ids = [13437,19132,12946,28186]

    # print(all_pred_data[0])
    pred_data = all_pred_data[0]
    true_spec_mzs = pred_data["spec_mzs"].numpy()
    true_spec_ints = pred_data["spec_ints"].numpy()
    true_spec_ints = true_spec_ints / np.linalg.norm(true_spec_ints, ord=1)
    true_spec_ints = true_spec_ints / np.sum(true_spec_ints)

    pred_spec_mzs = pred_data["pred_mzs"].numpy()
    pred_spec_ints = pred_data["pred_ints"].numpy()
    pred_spec_ints = pred_spec_ints / np.linalg.norm(pred_spec_ints, ord=1)
    # print(np.sum(pre_spec_ints))
    pred_spec_ints = pred_spec_ints / np.sum(pred_spec_ints)
    # print(pre_spec_ints)

    # cos_sim_bin = pred_data['cos_sim_0.01'].cpu()
    # cos_sim_hun = pred_data['cos_hun'].cpu()
    smiles = pred_data["smiles"][0]

    # check forumla logprobs
    pred_formula_logprobs = pred_data["pred_formula_logprobs"]
    pred_formula_probs = th.exp(pred_formula_logprobs)
    pred_formula_prob_idx = th.argsort(pred_formula_probs, descending=True)

    # node forumla p(n|f)
    pred_formula_node_logprobs = pred_data["pred_formula_node_logprobs"]

    # node prob index to node index mapping
    # node prob at i of pred_formula_node_logprobs belongs to node pred_joint_node_idxs[i]
    pred_joint_node_idxs = pred_data["pred_joint_node_idxs"]

    # node prob index to formula index mapping
    # node prob at i of pred_formula_node_logprobs belongs to forumla pred_joint_formula_idxs[i]
    pred_joint_formula_idxs = pred_data["pred_joint_formula_idxs"]

    # forumla prob index -> forumla index, forumla f[i] forumla i
    # forumla prob at i of pred_formula_logprobs is for formula pred_formula_formula_idxs[i]
    pred_formula_formula_idxs = pred_data["pred_formula_formula_idxs"]

    # forumla index -> peak index,  eg, forumla f[i] has peak i
    pred_frag_formula_peak_idxs = pred_data["frag_formula_peak_idxs"]
    # forumla index -> mz index , forumla at index i is mz j
    pred_frag_formula_peak_mzs = pred_data["frag_formula_peak_mzs"]

    # read frag from file
    frag_d = load_frag_d(13437, frag_dp, True)
    frag_cc_longs = frag_d["dag"].x[:, 5:6]
    num_nodes = frag_d["dag_num_nodes"]
    idx_to_formula = frag_d["idx_to_formula"]
    cc_bit_mask = th_long_to_mask(frag_cc_longs)

    # print(frag_d)
    num_formula = 5
    num_frags_per_formula = 3

    display_data = []
    peak_labels = {}
    mol = Chem.MolFromSmiles(smiles)

    display_row = 0
    peak_id = 1
    # for each formula
    for i, pf_idx in enumerate(pred_formula_prob_idx[0:num_formula]):
        # get formula idx
        f_idx = pred_formula_formula_idxs[pf_idx]

        bf_pred_formula_mask = pred_formula_formula_idxs == f_idx

        f_mz = pred_frag_formula_peak_mzs[pred_frag_formula_peak_idxs == f_idx].item()
        peak_idx = th.nonzero(pred_frag_formula_peak_idxs == f_idx).item()
        formula = idx_to_formula[f_idx.item()]
        f_pred_prob = pred_formula_probs[bf_pred_formula_mask].item()

        # in case peak is created by more then one forumla
        if peak_idx not in peak_labels:
            peak_labels[peak_idx] = peak_id
            peak_id += 1

        formula_data = {
            "formula_idx": f_idx.item(),
            "formula": formula,
            "formula_prob": f_pred_prob,
            "formula_mass": f_mz,
            "peak_idx": peak_idx,
            "peak_id": peak_labels[peak_idx],
            "frag": [],
        }

        print(
            "formula_idx:",
            f_idx.item(),
            "formula:",
            formula,
            "formula_prob:",
            f_pred_prob,
            "formula_mass:",
            f_mz,
            "peak_idx:",
            peak_idx,
        )

        bf_pred_formula_node_mask = pred_joint_formula_idxs == f_idx
        bf_pred_formula_node_logprobs = pred_formula_node_logprobs[bf_pred_formula_node_mask]
        bf_pred_formula_node_probs = th.exp(bf_pred_formula_node_logprobs)

        for pfn_idx in th.argsort(bf_pred_formula_node_probs, descending=True)[
            0:num_frags_per_formula
        ]:
            node_idx = pred_joint_node_idxs[bf_pred_formula_node_mask][pfn_idx]
            if bf_pred_formula_node_probs[pfn_idx] < 0.001:
                continue

            node_idx = node_idx.item()
            node_cc_bit_mask = cc_bit_mask[node_idx]

            node_atom_indices = node_cc_bit_mask.nonzero(as_tuple=True)[0].tolist()
            node_bond_indices = get_highlighted_bonds(mol, node_atom_indices)
            nf_pred_prob = bf_pred_formula_node_probs[pfn_idx].item()
            formula_data["frag"].append(
                {
                    "node_atom_indices": node_atom_indices,
                    "node_bond_indices": node_bond_indices,
                    "formula_node_prob": nf_pred_prob,
                }
            )
        display_data.append(formula_data)

    img = get_mol_im(smiles)
    w, h = img.size
    print(w, h)
    ratio = w / h
    print(w, h, ratio)

    right_ncols = 3 + 1
    right_rows = 5

    figsize = (30, 10)
    frag_col_w = figsize[1] / right_rows * 0.67 * ratio * right_ncols
    frag_col_w = int(ceil(frag_col_w)) + 2
    print(frag_col_w)

    fig = fig = plt.figure(figsize=figsize, dpi=300)
    grand_gs = GridSpec(ncols=figsize[0], nrows=1, figure=fig)
    left_gs = grand_gs[0 : figsize[0] - frag_col_w].subgridspec(ncols=1, nrows=1)
    right_gs = grand_gs[figsize[0] - frag_col_w :].subgridspec(ncols=right_ncols, nrows=right_rows)

    # spectrum ax
    ax = fig.add_subplot(left_gs[0], facecolor="white")
    plot_spectra_axes(
        ax,
        true_spec_mzs,
        true_spec_ints,
        pred_spec_mzs,
        pred_spec_ints,
        bar_width=1,
        font_size=24,
        true_ints_thresh=0,
        pred_ints_thresh=0,
        colors=["green", "red"],
        peak_labels=peak_labels,
    )
    # right side
    grid = [right_ncols, right_rows]
    for f_idx, formula_data in enumerate(display_data):
        # print(formula_data)
        # mol = formula_data['base_mol']
        formula = formula_data["formula"]
        formula_id = formula_data["formula"]
        peak_id = formula_data["peak_id"]
        ax = fig.add_subplot(right_gs[f_idx, 0])
        # formula_metrics = {"Formula Prob": formula_data['formula_prob'], "Mass" : formula_data['formula_mass']}
        plot_formula_axes(ax, formula, peak_id, formula_data["formula_prob"])
        for frag_idx, frag in enumerate(formula_data["frag"]):
            ax = fig.add_subplot(right_gs[f_idx, frag_idx + 1])
            h_atom_list = frag["node_atom_indices"]
            h_bond_list = frag["node_bond_indices"]
            # metrics = frag['metrics']
            # = []
            plot_mol_frag_axes(ax, mol, h_atom_list, h_bond_list, frag["formula_node_prob"])

    fig.savefig("./figs/annotation_test.png")
