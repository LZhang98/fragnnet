import pandas as pd
import os

fig_dp = "figs/entropy/jan_23_test"
fig_df = []

# sims
mean_sims_row = {"metric": "mean_cos_hun"}
cv_sims_row = {"metric": "cv_cos_hun"}
for ensemble in ["entropy", "baseline"]:
	spec_sims_df = pd.read_csv(os.path.join(fig_dp,f"{ensemble}_spec_sims_df.csv")).set_index("metric")
	mean_sims_row[ensemble] = spec_sims_df.loc["spec_sims_mean"]["mean"]
	cv_sims_row[ensemble] = spec_sims_df.loc["spec_sims_cv"]["mean"]
fig_df.append(mean_sims_row)
fig_df.append(cv_sims_row)

# entropy
for nb in ["", "nb_"]:
	for e in ["e", "ne"]:
		mean_h_row = {"metric": f"mean_{nb}{e}_fn"}
		cv_h_row = {"metric": f"cv_{nb}{e}_fn"}
		range_h_row = {"metric": f"range_{nb}{e}_fn"}
		for ensemble in ["entropy", "baseline"]:
			h_df = pd.read_csv(os.path.join(fig_dp,f"{ensemble}_{nb}h_df.csv")).set_index("entropy")
			h_df_row = h_df.loc[f"{nb}formula_node_{e}"]
			mean_h_row[ensemble] = h_df_row["mean"]
			cv_h_row[ensemble] = h_df_row["cv"]
			range_h_row[ensemble] = h_df_row["range"]
		fig_df.append(mean_h_row)
		fig_df.append(cv_h_row)
		fig_df.append(range_h_row)

# argmax
for nb in ["", "nb_"]:
	conses_am_row = {"metric": f"conses_{nb}am"}
	counts_am_row = {"metric": f"counts_{nb}am"}
	probs_am_row = {"metric": f"probs_{nb}am"}
	for ensemble in ["entropy", "baseline"]:
		am_df = pd.read_csv(os.path.join(fig_dp,f"{ensemble}_{nb}formula_node_argmax_df.csv")).set_index("metric")
		conses_am_row[ensemble] = am_df.loc[f"argmax_conses_tw"]["mean"]
		counts_am_row[ensemble] = am_df.loc[f"argmax_counts_tw"]["mean"]
		probs_am_row[ensemble] = am_df.loc[f"argmax_probs_tw"]["mean"]
	fig_df.append(conses_am_row)
	fig_df.append(counts_am_row)
	fig_df.append(probs_am_row)

fig_df = pd.DataFrame(fig_df)
print(fig_df)
fig_df.to_csv(os.path.join(fig_dp,"entropy_stats_df.csv"),index=False)


latex_df = fig_df.copy()
metrics = [
	"mean_cos_hun",
	"cv_cos_hun",
	"mean_ne_fn",
	"cv_ne_fn",
	"mean_nb_ne_fn",
	"cv_nb_ne_fn",
	"conses_am",
	"conses_nb_am",
	"probs_am",
	"probs_nb_am",
]
metric_to_name = {
	"mean_cos_hun": "Mean $cos_{\\textttt{HUN}}$",
	"cv_cos_hun": "CV $cos_{\\textttt{HUN}}$",
	"mean_ne_fn": "Mean $H_{\\theta}(n|f)$",
	"cv_ne_fn": "CV $H_{\\theta}(n|f)$",
	"mean_nb_ne_fn": "Mean $H_{\\theta}(\\tilde{n}|f)$",
	"cv_nb_ne_fn": "CV $H_{\\theta}(\\tilde{n}|f)$",
	"conses_am": "Cons $\\argmax_{n} P_{\\theta}(n|f)$",
	"conses_nb_am": "Cons $\\argmax_{\\tilde{n}} P_{\\theta}(\\tilde{n}|f)$",
	"probs_am": "Maj $\\argmax_{n} P_{\\theta}(n|f)$",
	"probs_nb_am": "Maj $\\argmax_{\\tilde{n}} P_{\\theta}(\\tilde{n}|f)$",
}
latex_df = latex_df[latex_df["metric"].isin(metrics)]
latex_df["metric"] = latex_df["metric"].apply(lambda x: metric_to_name[x])
for ensemble in ["entropy", "baseline"]:
	latex_df[ensemble] = latex_df[ensemble].apply(lambda x: f"${x:.3f}$")
latex_table = latex_df.to_latex(index=False, escape=False)
with open(os.path.join(fig_dp,"entropy_stats_df.tex"), "w") as f:
	f.write(latex_table)
print(latex_table)
import pdb; pdb.set_trace()


# fig_df = pd.DataFrame(fig_df)
# import pdb; pdb.set_trace()
# fig_df = fig_df.set_index("metric").transpose()
# fig_df["ensemble"] = fig_df.index
# fig_df = fig_df.reset_index(drop=True)
# print(fig_df)
# fig_df.to_csv(os.path.join(fig_dp,"entropy_stats_df.csv"),index=False)


# latex_df = fig_df.copy()
# cols = [
# 	"mean_cos_hun",
# 	"cv_cos_hun",
# 	"mean_ne_fn",
# 	"cv_ne_fn",
# 	"mean_nb_ne_fn",
# 	"cv_nb_ne_fn",
# 	"conses_am",
# 	"conses_nb_am",
# 	"probs_am",
# 	"probs_nb_am",
# ]
# latex_df = latex_df[["ensemble"]+cols]
# for col in cols:
# 	latex_df[col] = latex_df[col].apply(lambda x: f"{x:.3f}")
# latex_table = latex_df.to_latex(index=False, escape=False)
# with open(os.path.join(fig_dp,"entropy_stats_df.tex"), "w") as f:
# 	f.write(latex_table)
# print(latex_table)
# import pdb; pdb.set_trace()
