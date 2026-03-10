import logging
import os
import numpy as np
import pandas as pd

from generators import TokenGenerator, EmbeddingsGenerator
from preprocessing import annotate_pos, get_pos_distribution, get_pos_ngram_distribution, get_lexical_distribution
from preprocessing import export_pos_annotations, export_pos_annotations_relevant, export_pos_ngrams, export_pos_lexical

from drift import cos_drift, ks_drift, mmd_drift, js_drift, kl_drift, log_likelihood_drift


class MuSTDrifter:
    """ Multi-Source Temporal Drifter 
    """
    def __init__(self, df, df_name, results_path, n_jobs=20, K=100):
        self.df = df
        self.df_annotations= None
        self.df_name = df_name
        self.results_path = results_path
        self.encode= None

        self.n_jobs= n_jobs
        self.K= K
        
        self.pos_annotations_path=  f"{self.results_path}/{self.df_name}/data"

        self.pos_sintax_path=           f"{self.results_path}/{self.df_name}/data/sintax/all"
        self.pos_sintax_relevant_path=  f"{self.results_path}/{self.df_name}/data/sintax/relevant"
        self.pos_ngram_path=            f"{self.results_path}/{self.df_name}/data/sintax/ngram"
        self.pos_lexical_path=          f"{self.results_path}/{self.df_name}/data/lexical/relevant"
        self.embeddings_path=           f"{self.results_path}/{self.df_name}/data/semantic"

        self.sintax_drift_path=   f"{self.results_path}/{self.df_name}/drift/sintax"
        self.semantic_drift_path= f"{self.results_path}/{self.df_name}/drift/semantic"
        self.lexical_drift_path=  f"{self.results_path}/{self.df_name}/drift/lexical"
        
        os.makedirs(self.pos_annotations_path,     exist_ok=True)
        os.makedirs(self.pos_sintax_path,          exist_ok=True)
        os.makedirs(self.pos_sintax_relevant_path, exist_ok=True)
        os.makedirs(self.pos_ngram_path,           exist_ok=True)
        os.makedirs(self.pos_lexical_path,         exist_ok=True)
        os.makedirs(self.embeddings_path,          exist_ok=True)

        os.makedirs(self.sintax_drift_path,        exist_ok=True)
        os.makedirs(self.semantic_drift_path,      exist_ok=True)
        os.makedirs(self.lexical_drift_path,       exist_ok=True)


    
    def load_embeddings(self, period_id):
        with open(f'{self.embeddings_path}/{period_id}.npy', 'rb') as f:
            return np.load(f)

    def load_pos_lexical(self, period_id):
        with open(f'{self.pos_lexical_path}/{period_id}.npy', 'rb') as f:
            return np.load(f)
    
    def load_pos_ngram(self, period_id):
        with open(f'{self.pos_ngram_path}/{period_id}.npy', 'rb') as f:
            return np.load(f)
        
    def load_pos_sintax(self, period_id):
        with open(f'{self.pos_sintax_path}/{period_id}.npy', 'rb') as f:
            return np.load(f)

    def annotate_pos(self):
        self.df, self.df_annotations = annotate_pos(self.df, f"{self.pos_annotations_path}/dataset")

    def load_pos_annotation(self):
        self.df = pd.read_csv(f"{self.pos_annotations_path}/dataset.csv", index_col="Unnamed: 0")
        self.df_annotations= pd.read_csv(f"{self.pos_annotations_path}/dataset_pos.csv")
        return self.df, self.df_annotations

    def generate_pos_distributions(self):
        if self.df_annotations is None:
            raise ValueError("POS annotations not found. Please run annotate_pos() first.")
        self.df_pos_distribution=       get_pos_distribution(self.df_annotations)
        self.df_pos_ngram_distribution= get_pos_ngram_distribution(self.df_annotations)
        self.df_lexical_distribution=   get_lexical_distribution(self.df_annotations, self.df)

        export_pos_annotations(         self.df, self.df_pos_distribution,       filename_path=self.pos_sintax_path)
        export_pos_annotations_relevant(self.df, self.df_pos_distribution,       filename_path=self.pos_sintax_relevant_path)
        export_pos_ngrams(              self.df, self.df_pos_ngram_distribution, filename_path=self.pos_ngram_path, observed_pos=None)
        export_pos_lexical(                      self.df_lexical_distribution,   filename_path=self.pos_lexical_path)

    def generate_embeddings(self):
        if self.encode is None:
            self._init_encoder()
        for period_id, documents in self.df.groupby("period_id"):
            tokens= self.tokenize(documents["content"].dropna().astype(str).tolist())
            vectors= self.encode(tokens).numpy()
            with open(f'{self.embeddings_path}/{period_id}.npy', 'wb') as f:
                np.save(f, vectors)

    def generate_drift_dimensions(self):
        self.annotate_pos()
        self.generate_pos_distributions()
        self.generate_embeddings()

    def calculate_semantic_drift(self, reference_period, test_period, metrics=["cos_drift"]):
        reference_sample= self.load_embeddings(reference_period)
        test_sample= self.load_embeddings(test_period)

        filename=f"{self.semantic_drift_path}/{reference_period}_{test_period}"
        return self.calculate_drift(reference_sample=reference_sample, test_sample=test_sample, filename=filename, metrics=metrics)
        
    def calculate_sintactic_drift(self, reference_period, test_period, metrics=["cos_drift"]):
        reference_sample= self.load_pos_sintax(reference_period)
        test_sample= self.load_pos_sintax(test_period)

        filename=f"{self.sintax_drift_path}/{reference_period}_{test_period}"
        return self.calculate_drift(reference_sample=reference_sample, test_sample=test_sample, filename=filename, metrics=metrics)

    def calculate_lexical_drift(self, reference_period, test_period, metrics=["cos_drift"]):
        reference_sample= self.load_pos_lexical(reference_period)
        test_sample= self.load_pos_lexical(test_period)

        filename=f"{self.lexical_drift_path}/{reference_period}_{test_period}"
        return self.calculate_drift(reference_sample=reference_sample, test_sample=test_sample, filename=filename, metrics=metrics)

    def calculate_sintactic_ngram_drift(self, reference_period, test_period, metrics=["cos_drift"]):
        reference_sample= self.load_pos_ngram(reference_period)
        test_sample= self.load_pos_ngram(test_period)

        filename=f"{self.sintax_drift_path}/ngram_{reference_period}_{test_period}"
        return self.calculate_drift(reference_sample=reference_sample, test_sample=test_sample, filename=filename, metrics=metrics)

    def calculate_drift(self, reference_sample, test_sample, filename, metrics):
        drift= {}

        if "cos_drift" in metrics:
            drift["cos_drift"]= cos_drift(reference_sample=reference_sample, test_sample=test_sample, filename=f"{filename}_cos.json", K=self.K, n_jobs=self.n_jobs)
        if "ks_drift" in metrics:
            drift["ks_drift"]=  ks_drift( reference_sample=reference_sample, test_sample=test_sample, filename=f"{filename}_ks.json")
        if "mmd_drift" in metrics:
            drift["mmd_drift"]= mmd_drift(reference_sample=reference_sample, test_sample=test_sample, filename=f"{filename}_mmd.json", K=self.K, n_jobs=self.n_jobs)
        if "js_drift" in metrics:
            drift["js_drift"]= js_drift(reference_sample=reference_sample, test_sample=test_sample, filename=f"{filename}_js.json")
        if "kl_drift" in metrics:
            drift["kl_drift"]= kl_drift(reference_sample=reference_sample, test_sample=test_sample, filename=f"{filename}_kl.json")
        if "log_drift" in metrics:
            drift["log_drift"]= log_likelihood_drift(reference_sample=reference_sample, test_sample=test_sample, filename=f"{filename}_log.json", K=self.K, n_jobs=self.n_jobs)

        return drift

    def drift_analysis(self, reference_period, test_period, K=100):
        results= {}

        return results

    def all_drift_analysis(self):
        period_ids= self.df["period_id"].unique()
        results= {}
        for i in range(len(period_ids)-1):
            reference_period= period_ids[i]
            test_period= period_ids[i+1]
            results[f"{reference_period}_{test_period}"]= self.drift_analysis(reference_period, test_period, self.K)
        return results
    
    def _init_encoder(self, pretrained_model= "intfloat/multilingual-e5-large", tokenizer_max_len=512, device="cuda", batch_size=200):
        self.tokenizer=  TokenGenerator(pretrained_model, tokenizer_max_len=tokenizer_max_len ,batch_size=batch_size)
        self.tokenize=   self.tokenizer.tokenize_texts

        self.encoder=    EmbeddingsGenerator(pretrained_model, train_device=device, batch_size=batch_size)
        self.encode=     self.encoder.generate_embeddings
        
