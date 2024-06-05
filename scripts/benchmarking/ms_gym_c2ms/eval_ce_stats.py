import pandas as pd

if __name__ == "__main__":
    csv_fp = "model_eval/0jeuscvs.csv"
    df = pd.read_csv(csv_fp)
    print("cos_sim_hun by ev")
    by_ev_df = df.groupby('nce')['cos_sim_hun'].agg(['mean', 'std'])
    by_ev_df['mean'] = by_ev_df['mean'].round(3)
    by_ev_df.to_csv("model_eval/0jeuscvs_by_ev.csv")
    #print(df.groupby('nce')['cos_sim_bin'].agg(['mean', 'std']))
    #print("cos_sim_hun by inchikey")
    #print(df.groupby('inchikey')['cos_sim_hun'].agg(['mean']))
    
    csv_fp = "model_eval/bzh7mub7.csv"
    print("cos_sim_hun by ev")
    by_ev_df = df.groupby('nce')['cos_sim_hun'].agg(['mean', 'std'])
    by_ev_df['mean'] = by_ev_df['mean'].round(3)
    by_ev_df.to_csv("model_eval/bzh7mub7_by_ev.csv")
    #print(df.groupby('nce')['cos_sim_bin'].agg(['mean', 'std']))
    #print("cos_sim_hun by inchikey")
    #print(df.groupby('inchikey')['cos_sim_hun'].agg(['mean']))