import logging
import os
import numpy as np
import pandas as pd

import json

from .generators import TokenGenerator, EmbeddingsGenerator
from .preprocessing import annotate_pos, get_pos_distribution, get_pos_ngram_distribution, get_lexical_distribution, get_syntactic_content_distribution, get_syntactic_style_dimension
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

        self.syntactic_content_path=    f"{self.results_path}/{self.df_name}/data/syntax/content"
        self.syntactic_style_path=      f"{self.results_path}/{self.df_name}/data/syntax/style"
        self.lexical_path=              f"{self.results_path}/{self.df_name}/data/lexical"
        self.semantic_path=             f"{self.results_path}/{self.df_name}/data/semantic"
        self.thematic_path=             f"{self.results_path}/{self.df_name}/data/thematic"

        self.syntax_content_drift_path= f"{self.results_path}/{self.df_name}/drift/syntax/content"
        self.syntax_style_drift_path=   f"{self.results_path}/{self.df_name}/drift/syntax/style"
        self.lexical_drift_path=        f"{self.results_path}/{self.df_name}/drift/lexical"
        self.semantic_drift_path=       f"{self.results_path}/{self.df_name}/drift/semantic"
        self.thematic_drift_path=       f"{self.results_path}/{self.df_name}/drift/thematic"
        
        os.makedirs(self.pos_annotations_path,      exist_ok=True)
        os.makedirs(self.syntactic_content_path,    exist_ok=True)
        os.makedirs(self.syntactic_style_path,      exist_ok=True)
        os.makedirs(self.lexical_path,              exist_ok=True)
        os.makedirs(self.semantic_path,             exist_ok=True)
        os.makedirs(self.thematic_path,             exist_ok=True)

        os.makedirs(self.syntax_content_drift_path, exist_ok=True)
        os.makedirs(self.syntax_style_drift_path,   exist_ok=True)
        os.makedirs(self.lexical_drift_path,        exist_ok=True)
        os.makedirs(self.semantic_drift_path,       exist_ok=True)
        os.makedirs(self.thematic_drift_path,       exist_ok=True)

        # self.pos_sintax_path=           f"{self.results_path}/{self.df_name}/data/sintax/all"
        # self.pos_sintax_relevant_path=  f"{self.results_path}/{self.df_name}/data/sintax/relevant"
        # self.pos_ngram_path=            f"{self.results_path}/{self.df_name}/data/sintax/ngram"
        # self.pos_lexical_path=          f"{self.results_path}/{self.df_name}/data/lexical/relevant"
        # self.embeddings_path=           f"{self.results_path}/{self.df_name}/data/semantic"

        # self.syntax_drift_path=   f"{self.results_path}/{self.df_name}/drift/syntax"
        # self.semantic_drift_path= f"{self.results_path}/{self.df_name}/drift/semantic"
        # self.lexical_drift_path=  f"{self.results_path}/{self.df_name}/drift/lexical"
        
        # os.makedirs(self.pos_annotations_path,     exist_ok=True)
        # os.makedirs(self.pos_sintax_path,          exist_ok=True)
        # os.makedirs(self.pos_sintax_relevant_path, exist_ok=True)
        # os.makedirs(self.pos_ngram_path,           exist_ok=True)
        # os.makedirs(self.pos_lexical_path,         exist_ok=True)
        # os.makedirs(self.embeddings_path,          exist_ok=True)

        # os.makedirs(self.syntax_drift_path,        exist_ok=True)
        # os.makedirs(self.semantic_drift_path,      exist_ok=True)
        # os.makedirs(self.lexical_drift_path,       exist_ok=True)

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
        self.df, self.df_annotations = annotate_pos(self.df, f"{self.pos_annotations_path}/dataset", device=self.device)
        self.logger.info("POS annotation completed.")
    
    def _load_pos_annotation(self):
        self.logger.info("Loading POS annotations...")
        self.df = pd.read_csv(f"{self.pos_annotations_path}/dataset.csv", index_col="Unnamed: 0")
        self.df_annotations= pd.read_csv(f"{self.pos_annotations_path}/dataset_pos.csv")
        self.logger.info("POS annotations loaded.")
        return self.df, self.df_annotations

    def _require_pos_annotations(self):
        self.logger.debug("Checking for POS annotations...")
        if self.df_annotations is None:
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
    
    ### Loaders   
    def load_syntactic_content_dimension(self, period_id):
        _path= f'{self.syntactic_content_path}/{period_id}.npy'
        self.logger.debug(f"Syntax content dimension loaded from {_path}")
        with open(_path, 'rb') as f:
            return np.load(f)
    
    def load_syntactic_style_dimension(self, period_id):
        _path= f'{self.syntactic_style_path}/{period_id}.npy'
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
    def generate_syntactic_content_dimension(self):
        self.logger.info("Generating syntactic content dimension...")
        self.syntactic_distribution= get_syntactic_content_distribution(self.df_annotations, self.df)
        self.logger.debug("Syntactic content distribution generated.")
        
        pattern = [c for c in self.lexical_distribution.columns if c != "period_id"]
        
        with open(f"{self.syntactic_content_path}/dimensions_names.json", "w", encoding="utf-8") as f:
            json.dump(pattern, f, ensure_ascii=False)
            
        for _, row in self.syntactic_distribution.iterrows():
            period_id = row["period_id"].astype(int)
            vector = row[pattern].to_numpy(dtype=np.float64)

            with open(f"{self.syntactic_content_path}/{period_id}.npy", "wb") as f:
                np.save(f, vector)
    
    def generate_syntactic_style_dimension(self):
        self.logger.info("Generating syntactic style dimension...")
        self.syntactic_style_path= get_syntactic_style_dimension(self.df_annotations, self.df)
        self.logger.debug("Syntactic style distribution generated.")
        
        pattern= [c for c in self.syntactic_style_path.columns if c != "period_id"]
        
        with open(f"{self.syntactic_style_path}/dimensions_names.json", "w", encoding="utf-8") as f:
            json.dump(pattern, f, ensure_ascii=False)
            
        for _, row in self.syntactic_style_path.iterrows():
            period_id = row["period_id"].astype(int)
            vector = row[pattern].to_numpy(dtype=np.float64)

            with open(f"{self.syntactic_style_path}/{period_id}.npy", "wb") as f:
                np.save(f, vector)

    def generate_lexical_dimension(self):
        self.logger.info("Generating lexical dimension...")
        self._require_pos_annotations()
        
        self.lexical_distribution= get_lexical_distribution(self.df_annotations, self.df)
        
        self.logger.debug("Lexical distribution generated.")

        vocabulary = [c for c in self.lexical_distribution.columns if c != "period_id"]

        with open(f"{self.lexical_path}/dimensions_names.json", "w", encoding="utf-8") as f:
            json.dump(vocabulary, f, ensure_ascii=False)

        for _, row in self.lexical_distribution.iterrows():
            period_id = row["period_id"].astype(int)
            vector = row[vocabulary].to_numpy(dtype=np.float64)

            with open(f"{self.lexical_path}/{period_id}.npy", "wb") as f:
                np.save(f, vector)


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
        pass

    def generate_drift_dimensions(self):
        self.logger.info("Generating all dimensions for drift detection...")        
        self.generate_syntactic_content_dimension()
        self.generate_syntactic_style_dimension()
        self.generate_lexical_dimension()
        self.generate_semantic_dimension()
        self.generate_thematic_dimension()
    
    ###
    
    # def generate_pos_distributions(self):
    #     self.logger.info("Generating POS distributions...")

    #     if self.df_annotations is None:
    #         raise ValueError("POS annotations not found. Please run _annotate_pos() first.")

    #     self.df_pos_distribution= get_pos_distribution(self.df_annotations)
    #     self.logger.info("POS distribution generated.")

    #     self.df_pos_ngram_distribution= get_pos_ngram_distribution(self.df_annotations)
    #     self.logger.info("POS n-gram distribution generated.")

    #     self.df_lexical_distribution= get_lexical_distribution(self.df_annotations, self.df)
    #     self.logger.info("Lexical distribution generated.")

    #     export_pos_annotations(self.df, self.df_pos_distribution, filename_path=self.pos_syntax_path)
    #     self.logger.info("POS annotations exported for syntax.")

    #     # export_pos_annotations_relevant(self.df, self.df_pos_distribution, filename_path=self.pos_syntax_relevant_path)
    #     # self.logger.info("POS annotations exported for relevant syntax.")

    #     export_pos_ngrams(self.df, self.df_pos_ngram_distribution, filename_path=self.pos_ngram_path, observed_pos=None)
    #     self.logger.info("POS n-grams exported.")

    #     export_pos_lexical(self.df_lexical_distribution, filename_path=self.pos_lexical_path)
    #     self.logger.info("Lexical features exported.")


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
        reference_sample= self.load_syntactic_content_dimension(reference_period)
        test_sample= self.load_syntactic_content_dimension(test_period)

        filename=f"{self.syntax_content_drift_path}/{reference_period}_{test_period}"
        return self._calculate_drift(reference_sample=reference_sample, test_sample=test_sample, filename=filename, metrics=metrics, rebase=rebase)

    def calculate_syntactic_style_drift(self, reference_period, test_period, metrics=["js_drift", "kl_drift", "log_drift"], rebase=None):
        self.logger.info(f"Calculating syntactic drift between {reference_period} and {test_period} using metrics: {metrics}")
        reference_sample= self.load_syntactic_style_dimension(reference_period)
        test_sample= self.load_syntactic_style_dimension(test_period)

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