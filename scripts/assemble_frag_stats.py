import json
import pandas as pd

frag_param_to_fp = {
    "d=3,j=4": "data/frag/nist20_d3_h4_isoFalse_nb2/global_stats.json",
    "d=4,j=4": "data/frag/nist20_d4_h4_isoFalse_nb2/global_stats.json"
}

latex_df = []

stats = [
    "num_formulae",
    "dag_num_nodes",
    "dag_num_nodes_nb",
    "dag_num_edges",
    "recall_0.01_h4",
    "recall_10ppm_h4",
    "wrecall_0.01_h4",
    "wrecall_10ppm_h4",
]
stats_to_name = {
    "num_formulae": "\\# Formulae $f$",
    "dag_num_nodes": "\\# Nodes $n$",
    "dag_num_nodes_nb": "\\# Nodes $\\tilde{n}$",
    "dag_num_edges": "\\# Edges $e$",
    "recall_0.01_h4": "Recall (0.01 Da)",
    "recall_10ppm_h4": "Recall (10 ppm)",
    "wrecall_0.01_h4": "Weighted recall (0.01 Da)",
    "wrecall_10ppm_h4": "Weighted recall (10 ppm)",
}

for frag_param, fp in frag_param_to_fp.items():
    with open(fp, "r") as f:
        frag_stats = json.load(f)
    latex_df_entry = {"param": frag_param}
    for stat in stats:
        if stat.startswith("recall") or stat.startswith("wrecall"):
            stat_25p = float(frag_stats[f"m_spec/{stat}/25%"])
            stat_75p = float(frag_stats[f"m_spec/{stat}/75%"])
            latex_df_entry[stats_to_name[stat]] = f"${stat_25p:.3f} - {stat_75p:.3f}$"
        else:
            stat_25p = int(float(frag_stats[f"m_spec/{stat}/25%"]))
            stat_75p = int(float(frag_stats[f"m_spec/{stat}/75%"]))
            latex_df_entry[stats_to_name[stat]] = f"${stat_25p:d} - {stat_75p:d}$"
    latex_df.append(latex_df_entry)
latex_df = pd.DataFrame(latex_df)
latex_df = latex_df.set_index("param").transpose()
print(latex_df)
latex_df.to_csv("figs/frag/frag_stats_df.csv",index=True)
latex_table = latex_df.to_latex(index=True, escape=False)
print(latex_table)
with open("figs/frag/frag_stats_df.tex", "w") as f:
    f.write(latex_table)
import pdb; pdb.set_trace()

