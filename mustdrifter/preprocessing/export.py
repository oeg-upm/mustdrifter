import json
import numpy as np
import os

def export_pos_annotations(df, df_pos_distribution, filename_path, observed_pos=None):
    if observed_pos is not None:
        df_pos_distribution= df_pos_distribution[observed_pos+["doc_id"]]

    df_periods2doc_id= df.groupby("period_id").groups
    
    columns = [c for c in df_pos_distribution.columns if c != "period_id"]

    with open(f"{filename_path}/dimensions_names.json", "w", encoding="utf-8") as f:
        json.dump(columns, f, ensure_ascii=False)
        
    for period_id, doc_ids in df_periods2doc_id.items():
        vectors= df_pos_distribution[df_pos_distribution["doc_id"].isin(doc_ids.tolist())].drop("doc_id", axis=1)
        vectors= vectors.to_numpy()
        
        os.makedirs(filename_path, exist_ok=True)
        with open(f'{filename_path}/{period_id}.npy', 'wb') as f:
            np.save(f, vectors)

    return None

def export_pos_annotations_relevant(df, df_pos_distribution, filename_path, observed_pos=["NOUN", "VERB", "ADV", "ADJ"]):
    return export_pos_annotations(df, df_pos_distribution, filename_path, observed_pos)

def export_pos_ngrams(df, df_pos_distribution, filename_path, observed_pos=["NOUN+VERB", "NOUN+ADJ", "VERB+ADV", "PRON+PRON"]):
    return export_pos_annotations(df, df_pos_distribution, filename_path, observed_pos)

def export_pos_lexical(df_lexical_distribution, filename_path):
    vocabulary = sorted(c for c in df_lexical_distribution.columns if c != "period_id")

    with open(f"{filename_path}/dimensions_names.json", "w", encoding="utf-8") as f:
        json.dump(vocabulary, f, ensure_ascii=False)

    for _, row in df_lexical_distribution.iterrows():
        period_id = row["period_id"].astype(int)
        vector = row[vocabulary].to_numpy(dtype=np.float64)

        with open(f"{filename_path}/{period_id}.npy", "wb") as f:
            np.save(f, vector)
