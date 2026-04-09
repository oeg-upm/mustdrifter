import logging
import os
import numpy as np
import pandas as pd

import json

from .generators import TokenGenerator, EmbeddingsGenerator
from .preprocessing import annotate_pos, get_lexical_distribution, get_syntactic_content_distribution, get_syntactic_style_distribution, get_thematic_dimension
# from .preprocessing import get_pos_distribution, get_pos_ngram_distribution
# from .preprocessing import export_pos_annotations, export_pos_annotations_relevant, export_pos_ngrams, export_pos_lexical

from .drift import cos_drift, ks_drift, mmd_drift, js_drift, kl_drift, log_likelihood_drift

class MuSTDrifter:
    """ Multi-Source Temporal Drifter 
    """

    def __init__(self, df, df_name, results_path, n_jobs=20, K=100, device="cuda"):
        self.df = df
        self.df_name = df_name
        self.results_path = results_path
        self.encode= None

        self.device= device

        self.n_jobs= n_jobs
        self.K= K

        self.pos_annotations= None
        
        self.syntax_content_dimension= None
        self.syntax_style_dimension=   None
        self.lexical_dimension=        None
        self.semantic_dimension=       None
        self.thematic_dimension=       None        

        self.pos_annotations_path=      f"{self.results_path}/{self.df_name}/data"

        self.syntax_content_path=       f"{self.results_path}/{self.df_name}/data/syntax/content"
        self.syntax_style_path=         f"{self.results_path}/{self.df_name}/data/syntax/style"
        self.lexical_path=              f"{self.results_path}/{self.df_name}/data/lexical"
        self.semantic_path=             f"{self.results_path}/{self.df_name}/data/semantic"
        self.thematic_path=             f"{self.results_path}/{self.df_name}/data/thematic"

        self.syntax_content_drift_path= f"{self.results_path}/{self.df_name}/drift/syntax/content"
        self.syntax_style_drift_path=   f"{self.results_path}/{self.df_name}/drift/syntax/style"
        self.lexical_drift_path=        f"{self.results_path}/{self.df_name}/drift/lexical"
        self.semantic_drift_path=       f"{self.results_path}/{self.df_name}/drift/semantic"
        self.thematic_drift_path=       f"{self.results_path}/{self.df_name}/drift/thematic"
        
        os.makedirs(self.pos_annotations_path,      exist_ok=True)

        os.makedirs(self.syntax_content_path,    exist_ok=True)
        os.makedirs(self.syntax_style_path,      exist_ok=True)
        os.makedirs(self.lexical_path,              exist_ok=True)
        os.makedirs(self.semantic_path,             exist_ok=True)
        os.makedirs(self.thematic_path,             exist_ok=True)

        os.makedirs(self.syntax_content_drift_path, exist_ok=True)
        os.makedirs(self.syntax_style_drift_path,   exist_ok=True)
        os.makedirs(self.lexical_drift_path,        exist_ok=True)
        os.makedirs(self.semantic_drift_path,       exist_ok=True)
        os.makedirs(self.thematic_drift_path,       exist_ok=True)

        self.logger = logging.getLogger(__name__)

    ### Others
    def _init_encoder(self, pretrained_model= "intfloat/multilingual-e5-large", tokenizer_max_len=512, batch_size=200):
        self.tokenizer=  TokenGenerator(pretrained_model, tokenizer_max_len=tokenizer_max_len ,batch_size=batch_size)
        self.tokenize=   self.tokenizer.tokenize_texts

        self.encoder=    EmbeddingsGenerator(pretrained_model, train_device=self.device, batch_size=batch_size)
        self.encode=     self.encoder.generate_embeddings

        self.logger.info(f"Encoder initialized with model {pretrained_model} on device {self.device}.")

    def _annotate_pos(self):
        self.logger.info("Starting POS annotation...")
        self.df, self.pos_annotations = annotate_pos(self.df, f"{self.pos_annotations_path}/dataset", device=self.device)
        self.logger.info("POS annotation completed.")
    
    def _load_pos_annotation(self):
        self.logger.info("Loading POS annotations...")
        self.df = pd.read_csv(f"{self.pos_annotations_path}/dataset.csv", index_col="Unnamed: 0")
        self.pos_annotations= pd.read_csv(f"{self.pos_annotations_path}/dataset_pos.csv")
        self.logger.info("POS annotations loaded.")
        return self.df, self.pos_annotations

    def _require_pos_annotations(self):
        self.logger.debug("Checking for POS annotations...")
        if self.pos_annotations is None:
            try:
                self._load_pos_annotation()
            except Exception as e1:
                self.logger.warning("POS annotations not found. Attempting to annotate now...")
                try:
                    self._annotate_pos()
                except Exception as e2:
                    self.logger.error("Failed to annotate POS. Cannot generate lexical dimension.")
                    raise ValueError("POS annotations error.")
        return True
    ###
    
    ### Exporter
    def _export_dimension_annotations(self, df, filename_path):
        columns = [c for c in df.columns if c != "period_id"]

        with open(f"{filename_path}/dimensions_names.json", "w", encoding="utf-8") as f:
            json.dump(columns, f, ensure_ascii=False)

        for _, row in df.iterrows():
            period_id = row["period_id"].astype(int)
            vector = row[columns].to_numpy(dtype=np.float64)

            os.makedirs(filename_path, exist_ok=True)
            with open(f'{filename_path}/{period_id}.npy', 'wb') as f:
                np.save(f, vector)
    ###

    ### Loaders   
    def load_syntax_content_dimension(self, period_id):
        _path= f'{self.syntax_content_path}/{period_id}.npy'
        self.logger.debug(f"Syntax content dimension loaded from {_path}")
        with open(_path, 'rb') as f:
            return np.load(f)
    
    def load_syntax_style_dimension(self, period_id):
        _path= f'{self.syntax_style_path}/{period_id}.npy'
        self.logger.debug(f"Syntax style dimension loaded from {_path}")
        with open(_path, 'rb') as f:
            return np.load(f)
    
    def load_lexical_dimension(self, period_id):
        _path= f'{self.lexical_path}/{period_id}.npy'
        self.logger.debug(f"Lexical dimension loaded from {_path}")
        with open(_path, 'rb') as f:
            return np.load(f)

    def load_semantic_dimension(self, period_id):
        _path= f'{self.semantic_path}/{period_id}.npy'
        self.logger.debug(f"Semantic dimension loaded from {_path}")
        with open(_path, 'rb') as f:
            return np.load(f)
    
    def load_thematic_dimension(self, period_id):
        _path= f'{self.thematic_path}/{period_id}.npy'
        self.logger.debug(f"Thematic dimension loaded from {_path}")
        with open(_path, 'rb') as f:
            return np.load(f)
    ###
    
    ### Generators
    def generate_syntax_content_dimension(self):
        self.logger.info("Generating syntactic content dimension...")
        self._require_pos_annotations()

        self.syntax_content_dimension= get_syntactic_content_distribution(self.pos_annotations, self.df)
        self.logger.debug("Syntactic content distribution generated.")
        
        self._export_dimension_annotations(self.syntax_content_dimension, self.syntax_content_path)
        self.logger.info("Syntactic content features exported.")
        
    def generate_syntax_style_dimension(self):
        self.logger.info("Generating syntactic style dimension...")
        self._require_pos_annotations()

        self.syntax_style_dimension= get_syntactic_style_distribution(self.pos_annotations, self.df)
        self.logger.debug("Syntactic style distribution generated.")
        
        self._export_dimension_annotations(self.syntax_style_dimension, self.syntax_style_path)
        self.logger.info("Syntactic style features exported.")

    def generate_lexical_dimension(self):
        self.logger.info("Generating lexical dimension...")
        self._require_pos_annotations()
        
        self.lexical_dimension= get_lexical_distribution(self.pos_annotations, self.df)
        self.logger.debug("Lexical distribution generated.")

        self._export_dimension_annotations(self.lexical_dimension, self.lexical_path)
        self.logger.info("Lexical features exported.")

    def generate_semantic_dimension(self):
        self.logger.info("Generating embeddings...")
        if self.encode is None:
            self.logger.info("Encoder not initialized. Initializing now...")
            self._init_encoder()
            
        for period_id, documents in self.df.groupby("period_id"):
            tokens= self.tokenize(documents["content"].dropna().astype(str).tolist())
            vectors= self.encode(tokens).numpy()
            self.logger.debug(f"Embeddings generated for period {period_id} with shape {vectors.shape}.")

            with open(f'{self.semantic_path}/{period_id}.npy', 'wb') as f:
                np.save(f, vectors)

            self.logger.debug(f"Embeddings saved for period {period_id} at {self.semantic_path}/{period_id}.npy")

        self.logger.info("All embeddings generated and saved.")

    def generate_thematic_dimension(self):
        self.logger.info("Generating thematic dimension...")
        self.thematic_dimension= get_thematic_dimension(self.df)
        self.thematic_dimension = self.thematic_dimension.drop(columns=[-1, "-1"], errors="ignore")
        self.logger.debug("Thematic dimension generated.")
        
        self._export_dimension_annotations(self.thematic_dimension, self.thematic_path)
        self.logger.info("Thematic features exported.")

    def generate_drift_dimensions(self):
        self.logger.info("Generating all dimensions for drift detection...")        
        self.generate_syntax_content_dimension()
        self.generate_syntax_style_dimension()
        self.generate_lexical_dimension()
        self.generate_semantic_dimension()
        self.generate_thematic_dimension()
    ###

    ### Drift calculation
    def _calculate_drift(self, reference_sample, test_sample, filename, metrics, rebase=None):
        drift= {}

        if "cos_drift" in metrics:
            _filename= f"{filename}_cos.json"
            if (rebase is True) or (os.path.exists(_filename) and os.path.exists(_filename.replace(".json", "_bak.json")) and rebase is None) or (not os.path.exists(_filename)):
                drift["cos_drift"]= cos_drift(reference_sample=reference_sample, test_sample=test_sample, filename=_filename, K=self.K, n_jobs=self.n_jobs)
            else: 
                self.logger.info(f"Cosine drift result already exists at {_filename}. Skipping calculation.")
        if "ks_drift" in metrics:
            _filename= f"{filename}_ks.json"
            if (rebase is True) or (os.path.exists(_filename) and os.path.exists(_filename.replace(".json", "_bak.json")) and rebase is None) or (not os.path.exists(_filename)):
                drift["ks_drift"]=  ks_drift( reference_sample=reference_sample, test_sample=test_sample, filename=_filename)
            else:
                self.logger.info(f"KS drift result already exists at {_filename}. Skipping calculation.")
        if "mmd_drift" in metrics:
            _filename= f"{filename}_mmd.json"
            if (rebase is True) or (os.path.exists(_filename) and os.path.exists(_filename.replace(".json", "_bak.json")) and rebase is None) or (not os.path.exists(_filename)):
                drift["mmd_drift"]= mmd_drift(reference_sample=reference_sample, test_sample=test_sample, filename=_filename, K=self.K, n_jobs=self.n_jobs)
            else:
                self.logger.info(f"MMD drift result already exists at {_filename}. Skipping calculation.")
        if "js_drift" in metrics:
            _filename= f"{filename}_js.json"
            if (rebase is True) or (os.path.exists(_filename) and os.path.exists(_filename.replace(".json", "_bak.json")) and rebase is None) or (not os.path.exists(_filename)):
                drift["js_drift"]= js_drift(reference_sample=reference_sample, test_sample=test_sample, filename=_filename)
            else:
                self.logger.info(f"JS drift result already exists at {_filename}. Skipping calculation.")
        if "kl_drift" in metrics:
            _filename= f"{filename}_kl.json"
            if (rebase is True) or (os.path.exists(_filename) and os.path.exists(_filename.replace(".json", "_bak.json")) and rebase is None) or (not os.path.exists(_filename)):
                drift["kl_drift"]= kl_drift(reference_sample=reference_sample, test_sample=test_sample, filename=_filename)
            else:
                self.logger.info(f"KL drift result already exists at {_filename}. Skipping calculation.")
        if "log_drift" in metrics:
            _filename= f"{filename}_log.json"
            if (rebase is True) or (os.path.exists(_filename) and os.path.exists(_filename.replace(".json", "_bak.json")) and rebase is None) or (not os.path.exists(_filename)):
                drift["log_drift"]= log_likelihood_drift(reference_sample=reference_sample, test_sample=test_sample, filename=_filename, K=self.K, n_jobs=self.n_jobs)
            else:
                self.logger.info(f"Log-likelihood drift result already exists at {_filename}. Skipping calculation.")
        return drift

    def calculate_semantic_drift(self, reference_period, test_period, metrics=["cos_drift", "mmd_drift", "ks_drift"], rebase=None):
        self.logger.info(f"Calculating semantic drift between {reference_period} and {test_period} using metrics: {metrics}")
        reference_sample= self.load_semantic_dimension(reference_period)
        test_sample= self.load_semantic_dimension(test_period)

        filename=f"{self.semantic_drift_path}/{reference_period}_{test_period}"
        return self._calculate_drift(reference_sample=reference_sample, test_sample=test_sample, filename=filename, metrics=metrics, rebase=rebase)
        
    def calculate_syntactic_content_drift(self, reference_period, test_period, metrics=["js_drift", "kl_drift", "log_drift"], rebase=None):
        self.logger.info(f"Calculating syntactic drift between {reference_period} and {test_period} using metrics: {metrics}")
        reference_sample= self.load_syntax_content_dimension(reference_period)
        test_sample= self.load_syntax_content_dimension(test_period)

        filename=f"{self.syntax_content_drift_path}/{reference_period}_{test_period}"
        return self._calculate_drift(reference_sample=reference_sample, test_sample=test_sample, filename=filename, metrics=metrics, rebase=rebase)

    def calculate_syntactic_style_drift(self, reference_period, test_period, metrics=["js_drift", "kl_drift", "log_drift"], rebase=None):
        self.logger.info(f"Calculating syntactic drift between {reference_period} and {test_period} using metrics: {metrics}")
        reference_sample= self.load_syntax_style_dimension(reference_period)
        test_sample= self.load_syntax_style_dimension(test_period)

        filename=f"{self.syntax_style_drift_path}/{reference_period}_{test_period}"
        return self._calculate_drift(reference_sample=reference_sample, test_sample=test_sample, filename=filename, metrics=metrics, rebase=rebase)  

    def calculate_lexical_drift(self, reference_period, test_period, metrics=["js_drift", "kl_drift", "log_drift"], rebase=None):
        self.logger.info(f"Calculating lexical drift between {reference_period} and {test_period} using metrics: {metrics}")
        reference_sample= self.load_lexical_dimension(reference_period)
        test_sample= self.load_lexical_dimension(test_period)

        filename=f"{self.lexical_drift_path}/{reference_period}_{test_period}"
        return self._calculate_drift(reference_sample=reference_sample, test_sample=test_sample, filename=filename, metrics=metrics, rebase=rebase)

    def calculate_thematic_drift(self, reference_period, test_period, metrics=["js_drift", "kl_drift", "log_drift"], rebase=None):
        self.logger.info(f"Calculating thematic drift between {reference_period} and {test_period} using metrics: {metrics}")
        reference_sample= self.load_thematic_dimension(reference_period)
        test_sample= self.load_thematic_dimension(test_period)
        
        filename=f"{self.thematic_drift_path}/{reference_period}_{test_period}"
        return self._calculate_drift(reference_sample=reference_sample, test_sample=test_sample, filename=filename, metrics=metrics, rebase=rebase)

    def calculate_drift(self, drift_dimensions=["semantic", "syntactic_content", "syntactic_style", "lexical", "thematic"], metrics= None,rebase=None):
        self.logger.info("Calculating drift for all period pairs...")

        period_ids= self.df["period_id"].unique()

        for i in range(len(period_ids)-1):
            for e in range(len(period_ids)-1):
                if i == e: continue
                reference_period= period_ids[i]
                test_period= period_ids[e]
                self.logger.info(f"Starting drift calculation for period pair: {reference_period} vs {test_period}")

                if "semantic" in drift_dimensions:
                    if metrics is None:
                        _metrics= ["cos_drift", "mmd_drift", "ks_drift"]
                    else:
                        _metrics= metrics
                    self.calculate_semantic_drift( reference_period=reference_period, test_period=test_period, metrics=_metrics, rebase=rebase)

                if "syntactic_content" in drift_dimensions:
                    if metrics is None:
                        _metrics= ["js_drift", "kl_drift", "log_drift"]
                    else:
                        _metrics= metrics
                    self.calculate_syntactic_content_drift(reference_period=reference_period, test_period=test_period, metrics=_metrics, rebase=rebase)

                if "syntactic_style" in drift_dimensions:
                    if metrics is None:
                        _metrics= ["js_drift", "kl_drift", "log_drift"]
                    else:
                        _metrics= metrics
                    self.calculate_syntactic_style_drift(reference_period=reference_period, test_period=test_period, metrics=_metrics, rebase=rebase)

                if "lexical" in drift_dimensions:
                    if metrics is None:
                        _metrics= ["js_drift", "kl_drift", "log_drift"]
                    else:
                        _metrics= metrics
                    self.calculate_lexical_drift(reference_period=reference_period, test_period=test_period, metrics=_metrics, rebase=rebase)

                if "thematic" in drift_dimensions:
                    if metrics is None:
                        _metrics= ["js_drift", "kl_drift", "log_drift"]
                    else:
                        _metrics= metrics
                    self.calculate_thematic_drift(reference_period=reference_period, test_period=test_period, metrics=_metrics, rebase=rebase)

        self.logger.info("Drift calculation completed for all period pairs.")
    ###