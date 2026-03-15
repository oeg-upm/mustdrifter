import logging
import os
import numpy as np
import pandas as pd

from .generators import TokenGenerator, EmbeddingsGenerator
from .preprocessing import annotate_pos, get_pos_distribution, get_pos_ngram_distribution, get_lexical_distribution
from .preprocessing import export_pos_annotations, export_pos_annotations_relevant, export_pos_ngrams, export_pos_lexical

from .drift import cos_drift, ks_drift, mmd_drift, js_drift, kl_drift, log_likelihood_drift

class MuSTDrifter:
    """ Multi-Source Temporal Drifter 
    """
    def __init__(self, df, df_name, results_path, n_jobs=20, K=100, device="cuda"):
        self.df = df
        self.df_annotations= None
        self.df_name = df_name
        self.results_path = results_path
        self.encode= None

        self.device= device
        
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

        self.logger = logging.getLogger(__name__)

    def _init_encoder(self, pretrained_model= "intfloat/multilingual-e5-large", tokenizer_max_len=512, batch_size=200):
        self.tokenizer=  TokenGenerator(pretrained_model, tokenizer_max_len=tokenizer_max_len ,batch_size=batch_size)
        self.tokenize=   self.tokenizer.tokenize_texts

        self.encoder=    EmbeddingsGenerator(pretrained_model, train_device=self.device, batch_size=batch_size)
        self.encode=     self.encoder.generate_embeddings

        self.logger.info(f"Encoder initialized with model {pretrained_model} on device {self.device}.")

    def load_embeddings(self, period_id):
        _path= f'{self.embeddings_path}/{period_id}.npy'
        with open(_path, 'rb') as f:
            return np.load(f)
        self.logger.debug(f"Embeddings loaded from {period_id} at {_path}")

    def load_pos_lexical(self, period_id):
        _path= f'{self.pos_lexical_path}/{period_id}.npy'
        with open(_path, 'rb') as f:
            return np.load(f)
        self.logger.debug(f"Lexical features loaded from {period_id} at {_path}")
        
    def load_pos_ngram(self, period_id):
        _path= f'{self.pos_ngram_path}/{period_id}.npy'
        with open(_path, 'rb') as f:
            return np.load(f)
        self.logger.debug(f"N-gram features loaded from {period_id} at {_path}")
        
    def load_pos_sintax(self, period_id):
        _path= f'{self.pos_sintax_path}/{period_id}.npy'
        with open(_path, 'rb') as f:
            return np.load(f)
        self.logger.debug(f"Sintax features loaded from {period_id} at {_path}")

    def annotate_pos(self):
        self.logger.info("Starting POS annotation...")
        self.df, self.df_annotations = annotate_pos(self.df, f"{self.pos_annotations_path}/dataset", device=self.device)
        self.logger.info("POS annotation completed.")
    
    def load_pos_annotation(self):
        self.logger.info("Loading POS annotations...")
        self.df = pd.read_csv(f"{self.pos_annotations_path}/dataset.csv", index_col="Unnamed: 0")
        self.df_annotations= pd.read_csv(f"{self.pos_annotations_path}/dataset_pos.csv")
        self.logger.info("POS annotations loaded.")
        return self.df, self.df_annotations

    def generate_pos_distributions(self):
        self.logger.info("Generating POS distributions...")

        if self.df_annotations is None:
            raise ValueError("POS annotations not found. Please run annotate_pos() first.")

        self.df_pos_distribution=       get_pos_distribution(self.df_annotations)
        self.logger.info("POS distribution generated.")

        self.df_pos_ngram_distribution= get_pos_ngram_distribution(self.df_annotations)
        self.logger.info("POS n-gram distribution generated.")

        self.df_lexical_distribution=   get_lexical_distribution(self.df_annotations, self.df)
        self.logger.info("Lexical distribution generated.")

        export_pos_annotations(         self.df, self.df_pos_distribution,       filename_path=self.pos_sintax_path)
        self.logger.info("POS annotations exported for sintax.")

        export_pos_annotations_relevant(self.df, self.df_pos_distribution,       filename_path=self.pos_sintax_relevant_path)
        self.logger.info("POS annotations exported for relevant sintax.")

        export_pos_ngrams(              self.df, self.df_pos_ngram_distribution, filename_path=self.pos_ngram_path, observed_pos=None)
        self.logger.info("POS n-grams exported.")

        export_pos_lexical(                      self.df_lexical_distribution,   filename_path=self.pos_lexical_path)
        self.logger.info("Lexical features exported.")

    def generate_embeddings(self):
        self.logger.info("Generating embeddings...")
        if self.encode is None:
            self.logger.info("Encoder not initialized. Initializing now...")
            self._init_encoder()
            
        for period_id, documents in self.df.groupby("period_id"):
            tokens= self.tokenize(documents["content"].dropna().astype(str).tolist())
            vectors= self.encode(tokens).numpy()
            self.logger.info(f"Embeddings generated for period {period_id} with shape {vectors.shape}.")

            with open(f'{self.embeddings_path}/{period_id}.npy', 'wb') as f:
                np.save(f, vectors)

            self.logger.debug(f"Embeddings saved for period {period_id} at {self.embeddings_path}/{period_id}.npy")

        self.logger.info("All embeddings generated and saved.")
        
    def generate_drift_dimensions(self):
        self.logger.info("Generating dimensions for drift detection...")
        self.annotate_pos()
        self.generate_pos_distributions()
        self.generate_embeddings()

    def calculate_drift(self, reference_sample, test_sample, filename, metrics, rebase=False):
        drift= {}

        if "cos_drift" in metrics:
            _filename= f"{filename}_cos.json"
            if not os.path.exists(_filename) or rebase or os.path.exists(_filename.replace(".json", "_bak.json")):
                drift["cos_drift"]= cos_drift(reference_sample=reference_sample, test_sample=test_sample, filename=_filename, K=self.K, n_jobs=self.n_jobs)
            else: 
                self.logger.info(f"Cosine drift result already exists at {_filename}. Skipping calculation.")
        if "ks_drift" in metrics:
            _filename= f"{filename}_ks.json"
            if not os.path.exists(_filename) or rebase or os.path.exists(_filename.replace(".json", "_bak.json")):
                drift["ks_drift"]=  ks_drift( reference_sample=reference_sample, test_sample=test_sample, filename=_filename)
            else:
                self.logger.info(f"KS drift result already exists at {_filename}. Skipping calculation.")
        if "mmd_drift" in metrics:
            _filename= f"{filename}_mmd.json"
            if not os.path.exists(_filename) or rebase or os.path.exists(_filename.replace(".json", "_bak.json")):
                drift["mmd_drift"]= mmd_drift(reference_sample=reference_sample, test_sample=test_sample, filename=_filename, K=self.K, n_jobs=self.n_jobs)
            else:
                self.logger.info(f"MMD drift result already exists at {_filename}. Skipping calculation.")
        if "js_drift" in metrics:
            _filename= f"{filename}_js.json"
            if not os.path.exists(_filename) or rebase or os.path.exists(_filename.replace(".json", "_bak.json")):
                drift["js_drift"]= js_drift(reference_sample=reference_sample, test_sample=test_sample, filename=_filename)
            else:
                self.logger.info(f"JS drift result already exists at {_filename}. Skipping calculation.")
        if "kl_drift" in metrics:
            _filename= f"{filename}_kl.json"
            if not os.path.exists(_filename) or rebase or os.path.exists(_filename.replace(".json", "_bak.json")):
                drift["kl_drift"]= kl_drift(reference_sample=reference_sample, test_sample=test_sample, filename=_filename)
            else:
                self.logger.info(f"KL drift result already exists at {_filename}. Skipping calculation.")
        if "log_drift" in metrics:
            _filename= f"{filename}_log.json"
            if not os.path.exists(_filename) or rebase or os.path.exists(_filename.replace(".json", "_bak.json")):
                drift["log_drift"]= log_likelihood_drift(reference_sample=reference_sample, test_sample=test_sample, filename=_filename, K=self.K, n_jobs=self.n_jobs)
            else:
                self.logger.info(f"Log-likelihood drift result already exists at {_filename}. Skipping calculation.")
        return drift

    def calculate_semantic_drift(self, reference_period, test_period, metrics=["cos_drift", "mmd_drift", "ks_drift"], rebase=False):
        self.logger.info(f"Calculating semantic drift between {reference_period} and {test_period} using metrics: {metrics}")
        reference_sample= self.load_embeddings(reference_period)
        test_sample= self.load_embeddings(test_period)

        filename=f"{self.semantic_drift_path}/{reference_period}_{test_period}"
        return self.calculate_drift(reference_sample=reference_sample, test_sample=test_sample, filename=filename, metrics=metrics, rebase=rebase)
        
    def calculate_sintactic_drift(self, reference_period, test_period, metrics=["js_drift", "kl_drift", "log_drift"], rebase=False):
        self.logger.info(f"Calculating sintactic drift between {reference_period} and {test_period} using metrics: {metrics}")
        reference_sample= self.load_pos_sintax(reference_period)
        test_sample= self.load_pos_sintax(test_period)

        filename=f"{self.sintax_drift_path}/{reference_period}_{test_period}"
        return self.calculate_drift(reference_sample=reference_sample, test_sample=test_sample, filename=filename, metrics=metrics, rebase=rebase)

    def calculate_lexical_drift(self, reference_period, test_period, metrics=["js_drift", "kl_drift", "log_drift"], rebase=False):
        self.logger.info(f"Calculating lexical drift between {reference_period} and {test_period} using metrics: {metrics}")
        reference_sample= self.load_pos_lexical(reference_period)
        test_sample= self.load_pos_lexical(test_period)

        filename=f"{self.lexical_drift_path}/{reference_period}_{test_period}"
        return self.calculate_drift(reference_sample=reference_sample, test_sample=test_sample, filename=filename, metrics=metrics, rebase=rebase)

    def calculate_sintactic_ngram_drift(self, reference_period, test_period, metrics=["js_drift", "kl_drift", "log_drift"], rebase=False):
        self.logger.info(f"Calculating sintactic n-gram drift between {reference_period} and {test_period} using metrics: {metrics}")
        reference_sample= self.load_pos_ngram(reference_period)
        test_sample= self.load_pos_ngram(test_period)

        filename=f"{self.sintax_drift_path}/ngram_{reference_period}_{test_period}"
        return self.calculate_drift(reference_sample=reference_sample, test_sample=test_sample, filename=filename, metrics=metrics, rebase=rebase)

    def calculate_all_drift(self, drift_dimensions=["semantic", "sintactic", "lexical"], rebase=False):
        self.logger.info("Calculating drift for all period pairs...")

        period_ids= self.df["period_id"].unique()

        for i in range(len(period_ids)-1):
            for e in range(len(period_ids)-1):
                if i == e: continue
                reference_period= period_ids[i]
                test_period= period_ids[e]
                self.logger.info(f"Starting drift calculation for period pair: {reference_period} vs {test_period}")

                if "semantics" in drift_dimensions:
                    self.calculate_semantic_drift( reference_period=reference_period, test_period=test_period, metrics=["cos_drift", "mmd_drift", "ks_drift"],           rebase=rebase)
                if "sintactic" in drift_dimensions:
                    self.calculate_sintactic_drift(reference_period=reference_period, test_period=test_period, metrics=["js_drift", "kl_drift", "log_likelihood_drift"], rebase=rebase)
                if "lexical" in drift_dimensions:
                    self.calculate_lexical_drift(  reference_period=reference_period, test_period=test_period, metrics=["js_drift", "kl_drift", "log_likelihood_drift"], rebase=rebase)

        self.logger.info("Drift calculation completed for all period pairs.")

