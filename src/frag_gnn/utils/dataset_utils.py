import pandas as pd


from frag_gnn.utils.spec_utils import merge_sparse_specs


def merge_spec_df(spec_df):

	# merge the peaks (they are not normalized here)
	m_func = lambda peakses: merge_sparse_specs(*peakses,renormalize=False)
	m_spec_df = spec_df[["group_id","peaks"]].groupby("group_id").agg({"peaks":m_func}).reset_index()
	# add the metadata back
	meta_spec_df = spec_df.drop(columns=["peaks","spec_id"]).drop_duplicates(subset=["group_id"])
	m_spec_df = m_spec_df.merge(meta_spec_df,on=["group_id"],how="inner")
	return m_spec_df