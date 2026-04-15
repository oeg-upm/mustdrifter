import logging
import os
import numpy as np
import pandas as pd
import re
import json
from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns

from .generators import TokenGenerator, EmbeddingsGenerator
from .preprocessing import annotate_pos
from .preprocessing import get_lexical_distribution, get_syntactic_content_distribution, get_syntactic_style_distribution, get_syntactic_style_sub_distributions, get_thematic_dimension

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

        self.report_drift_tables=      None

        self.base_path=                 f"{self.results_path}/{self.df_name}"

        self.data_path=      f"{self.base_path}/data"
        self.drift_path=     f"{self.base_path}/drift"
        self.report_path=    f"{self.base_path}/report"

        self.syntax_content_data_path=       f"{self.data_path}/syntax/content"
        self.syntax_style_data_path=         f"{self.data_path}/syntax/style"
        self.lexical_data_path=              f"{self.data_path}/lexical"
        self.semantic_data_path=             f"{self.data_path}/semantic"
        self.thematic_data_path=             f"{self.data_path}/thematic"

        self.syntax_content_drift_path= f"{self.drift_path}/syntax/content"
        self.syntax_style_drift_path=   f"{self.drift_path}/syntax/style"
        self.lexical_drift_path=        f"{self.drift_path}/lexical"
        self.semantic_drift_path=       f"{self.drift_path}/semantic"
        self.thematic_drift_path=       f"{self.drift_path}/thematic"

        os.makedirs(self.base_path,   exist_ok=True)
        os.makedirs(self.data_path,   exist_ok=True)
        os.makedirs(self.drift_path,  exist_ok=True)
        os.makedirs(self.report_path, exist_ok=True)

        os.makedirs(self.syntax_content_data_path, exist_ok=True)
        os.makedirs(self.syntax_style_data_path,   exist_ok=True)
        os.makedirs(self.lexical_data_path,        exist_ok=True)
        os.makedirs(self.semantic_data_path,       exist_ok=True)
        os.makedirs(self.thematic_data_path,       exist_ok=True)

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
        self.df, self.pos_annotations = annotate_pos(self.df, f"{self.data_path}/dataset", device=self.device)
        self.logger.info("POS annotation completed.")
    
    def _load_pos_annotation(self):
        self.logger.debug("Loading POS annotations...")
        self.df = pd.read_csv(f"{self.data_path}/dataset.csv", index_col="Unnamed: 0")
        self.pos_annotations= pd.read_csv(f"{self.data_path}/dataset_pos.csv")
        self.logger.debug("POS annotations loaded.")
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

    ### Dimension Loaders   
    def load_syntax_content_dimension(self, period_id):
        _path= f'{self.syntax_content_data_path}/{period_id}.npy'
        self.logger.debug(f"Syntax content dimension loaded from {_path}")
        with open(_path, 'rb') as f:
            return np.load(f)
    
    def load_syntax_style_dimension(self, period_id):
        _path= f'{self.syntax_style_data_path}/{period_id}.npy'
        self.logger.debug(f"Syntax style dimension loaded from {_path}")
        with open(_path, 'rb') as f:
            return np.load(f)
    
    def load_lexical_dimension(self, period_id):
        _path= f'{self.lexical_data_path}/{period_id}.npy'
        self.logger.debug(f"Lexical dimension loaded from {_path}")
        with open(_path, 'rb') as f:
            return np.load(f)

    def load_semantic_dimension(self, period_id):
        _path= f'{self.semantic_data_path}/{period_id}.npy'
        self.logger.debug(f"Semantic dimension loaded from {_path}")
        with open(_path, 'rb') as f:
            return np.load(f)
    
    def load_thematic_dimension(self, period_id):
        _path= f'{self.thematic_data_path}/{period_id}.npy'
        self.logger.debug(f"Thematic dimension loaded from {_path}")
        with open(_path, 'rb') as f:
            return np.load(f)
    
    def load_dimension_names(self, dimension):
        dimension_names_file= "dimensions_names.json"
        if dimension == "syntax_content":
            path= f"{self.syntax_content_data_path}/{dimension_names_file}"
        elif dimension == "syntax_style":
            path= f"{self.syntax_style_data_path}/{dimension_names_file}"
        elif dimension == "lexical":
            path= f"{self.lexical_data_path}/{dimension_names_file}"
        elif dimension == "semantic":
            path= f"{self.semantic_data_path}/{dimension_names_file}"
        elif dimension == "thematic":
            path= f"{self.thematic_data_path}/{dimension_names_file}"
        else:
            raise ValueError(f"Unknown dimension: {dimension}")

        self.logger.debug(f"Loading dimension names for {dimension} from {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    ###
    
    ### Drift Loaders
    
    def load_syntax_content_drift(self, reference_period, test_period, metric):
        _path = f"{self.syntax_content_drift_path}/{reference_period}_{test_period}_{metric}.json"
        self.logger.debug(f"Syntax content drift loaded from {_path}")
        with open(_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_syntax_style_drift(self, reference_period, test_period, metric):
        _path = f"{self.syntax_style_drift_path}/{reference_period}_{test_period}_{metric}.json"
        self.logger.debug(f"Syntax style drift loaded from {_path}")
        with open(_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_lexical_drift(self, reference_period, test_period, metric):
        _path = f"{self.lexical_drift_path}/{reference_period}_{test_period}_{metric}.json"
        self.logger.debug(f"Lexical drift loaded from {_path}")
        with open(_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_semantic_drift(self, reference_period, test_period, metric):
        _path = f"{self.semantic_drift_path}/{reference_period}_{test_period}_{metric}.json"
        self.logger.debug(f"Semantic drift loaded from {_path}")
        with open(_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_thematic_drift(self, reference_period, test_period, metric):
        _path = f"{self.thematic_drift_path}/{reference_period}_{test_period}_{metric}.json"
        self.logger.debug(f"Thematic drift loaded from {_path}")
        with open(_path, "r", encoding="utf-8") as f:
            return json.load(f)
    ###
    
    ### Generators
    def generate_syntax_content_dimension(self, **kwargs):
        self.logger.info("Generating syntactic content dimension...")
        self._require_pos_annotations()

        self.syntax_content_dimension= get_syntactic_content_distribution(self.pos_annotations, self.df, **kwargs)
        self.logger.debug("Syntactic content distribution generated.")
        
        self._export_dimension_annotations(self.syntax_content_dimension, self.syntax_content_data_path)
        self.logger.info("Syntactic content features exported.")
        
    def generate_syntax_style_dimension(self, **kwargs):
        self.logger.info("Generating syntactic style dimension...")
        self._require_pos_annotations()

        self.syntax_style_dimension= get_syntactic_style_distribution(self.pos_annotations, self.df, **kwargs)
        self.logger.debug("Syntactic style distribution generated.")
        
        self._export_dimension_annotations(self.syntax_style_dimension, self.syntax_style_data_path)
        self.logger.info("Syntactic style features exported.")

    def generate_lexical_dimension(self, **kwargs):
        self.logger.info("Generating lexical dimension...")
        self._require_pos_annotations()
        
        self.lexical_dimension= get_lexical_distribution(self.pos_annotations, self.df, **kwargs)
        self.logger.debug("Lexical distribution generated.")

        self._export_dimension_annotations(self.lexical_dimension, self.lexical_data_path)
        self.logger.info("Lexical features exported.")

    def generate_semantic_dimension(self, **kwargs):
        self.logger.info("Generating embeddings...")
        if self.encode is None:
            self.logger.debug("Encoder not initialized. Initializing now...")
            self._init_encoder()
            
        for period_id, documents in self.df.groupby("period_id"):
            tokens= self.tokenize(documents["content"].dropna().astype(str).tolist())
            vectors= self.encode(tokens).numpy()
            self.logger.debug(f"Embeddings generated for period {period_id} with shape {vectors.shape}.")

            with open(f'{self.semantic_data_path}/{period_id}.npy', 'wb') as f:
                np.save(f, vectors)

            self.logger.debug(f"Embeddings saved for period {period_id} at {self.semantic_data_path}/{period_id}.npy")

        self.logger.info("All embeddings generated and saved.")

    def generate_thematic_dimension(self, **kwargs):
        self.logger.info("Generating thematic dimension...")
        self.thematic_dimension= get_thematic_dimension(self.df, **kwargs)
        self.thematic_dimension = self.thematic_dimension.drop(columns=[-1, "-1"], errors="ignore")
        self.logger.debug("Thematic dimension generated.")
        
        self._export_dimension_annotations(self.thematic_dimension, self.thematic_data_path)
        self.logger.info("Thematic features exported.")

    def generate_drift_dimensions(self, dimensions=["semantic", "syntactic_content", "syntactic_style", "lexical", "thematic"], **kwargs):
        self.logger.info(f"Generating {dimensions} dimensions for drift detection...")

        if "syntactic_content" in dimensions:
            self.generate_syntax_content_dimension(**kwargs)
            self.logger.debug("Syntactic content dimension generated and exported.")

        if "syntactic_style" in dimensions:
            self.generate_syntax_style_dimension(**kwargs)
            self.logger.debug("Syntactic style dimension generated and exported.")
             
        if "lexical" in dimensions:
            self.generate_lexical_dimension(**kwargs)
            self.logger.debug("Lexical dimension generated and exported.")
            
        if "semantic" in dimensions:
            self.generate_semantic_dimension(**kwargs)
            self.logger.debug("Semantic dimension generated and exported.")
             
        if "thematic" in dimensions:
            self.generate_thematic_dimension(**kwargs)
            self.logger.debug("Thematic dimension generated and exported.")

    ###

    ### Drift calculation
    def _calculate_drift(self, reference_sample, test_sample, filename, metrics, rebase=None):
        drift= {}

        if "cos_drift" in metrics:
            _filename= f"{filename}_cos.json"
            if ((rebase is True) or 
                (os.path.exists(_filename) and os.path.exists(_filename.replace(".json", "_bak.json")) and rebase is None) or
                (not os.path.exists(_filename))
            ):
                drift["cos_drift"]= cos_drift(reference_sample=reference_sample, test_sample=test_sample, filename=_filename, K=self.K, n_jobs=self.n_jobs)
            else: 
                self.logger.debug(f"Cosine drift result already exists at {_filename}. Skipping calculation.")

        if "ks_drift" in metrics:
            _filename= f"{filename}_ks.json"
            if ((rebase is True) or 
                (os.path.exists(_filename) and os.path.exists(_filename.replace(".json", "_bak.json")) and rebase is None) or 
                (not os.path.exists(_filename))
            ):
                drift["ks_drift"]=  ks_drift( reference_sample=reference_sample, test_sample=test_sample, filename=_filename)
            else:
                self.logger.debug(f"KS drift result already exists at {_filename}. Skipping calculation.")

        if "mmd_drift" in metrics:
            _filename= f"{filename}_mmd.json"
            if ((rebase is True) or 
                (os.path.exists(_filename) and os.path.exists(_filename.replace(".json", "_bak.json")) and rebase is None) or 
                (not os.path.exists(_filename))
            ):
                drift["mmd_drift"]= mmd_drift(reference_sample=reference_sample, test_sample=test_sample, filename=_filename, K=self.K, n_jobs=self.n_jobs)
            else:
                self.logger.debug(f"MMD drift result already exists at {_filename}. Skipping calculation.")

        if "js_drift" in metrics:
            _filename= f"{filename}_js.json"
            if filename is None: _filename = ""
            if ((rebase is True) or 
                (os.path.exists(_filename) and os.path.exists(_filename.replace(".json", "_bak.json")) and rebase is None) or 
                (not os.path.exists(_filename)) or 
                (_filename == "")
            ):
                drift["js_drift"]= js_drift(reference_sample=reference_sample, test_sample=test_sample, filename=_filename)
            else:
                self.logger.debug(f"JS drift result already exists at {_filename}. Skipping calculation.")

        if "kl_drift" in metrics:
            _filename= f"{filename}_kl.json"
            if filename is None: _filename = ""
            if ((rebase is True) or 
                (os.path.exists(_filename) and os.path.exists(_filename.replace(".json", "_bak.json")) and rebase is None) or 
                (not os.path.exists(_filename)) or 
                (_filename == "")
            ):
                drift["kl_drift"]= kl_drift(reference_sample=reference_sample, test_sample=test_sample, filename=_filename)
            else:
                self.logger.debug(f"KL drift result already exists at {_filename}. Skipping calculation.")

        if "log_drift" in metrics:
            _filename= f"{filename}_log.json"
            if filename is None: _filename = ""
            if ((rebase is True) or 
                (os.path.exists(_filename) and os.path.exists(_filename.replace(".json", "_bak.json")) and rebase is None) or 
                (not os.path.exists(_filename)) or 
                (_filename == "")
            ):
                drift["log_drift"]= log_likelihood_drift(reference_sample=reference_sample, test_sample=test_sample, filename=_filename, K=self.K, n_jobs=self.n_jobs)
            else:
                self.logger.debug(f"Log-likelihood drift result already exists at {_filename}. Skipping calculation.")

        return drift

    def calculate_semantic_drift(self, reference_period, test_period, metrics=["cos_drift", "mmd_drift", "ks_drift"], rebase=None):
        self.logger.debug(f"Calculating semantic drift between {reference_period} and {test_period} using metrics: {metrics}")
        reference_sample= self.load_semantic_dimension(reference_period)
        test_sample= self.load_semantic_dimension(test_period)

        filename=f"{self.semantic_drift_path}/{reference_period}_{test_period}"
        return self._calculate_drift(reference_sample=reference_sample, test_sample=test_sample, filename=filename, metrics=metrics, rebase=rebase)
        
    def calculate_syntactic_content_drift(self, reference_period, test_period, metrics=["js_drift", "kl_drift", "log_drift"], rebase=None):
        self.logger.debug(f"Calculating syntactic content drift between {reference_period} and {test_period} using metrics: {metrics}")
        reference_sample= self.load_syntax_content_dimension(reference_period)
        test_sample= self.load_syntax_content_dimension(test_period)

        filename=f"{self.syntax_content_drift_path}/{reference_period}_{test_period}"
        return self._calculate_drift(reference_sample=reference_sample, test_sample=test_sample, filename=filename, metrics=metrics, rebase=rebase)

    def calculate_syntactic_style_drift(self, reference_period, test_period, metrics=["js_drift", "kl_drift", "log_drift"], rebase=None):
        self.logger.debug(f"Calculating syntactic style drift between {reference_period} and {test_period} using metrics: {metrics}")
        reference_sample= self.load_syntax_style_dimension(reference_period)
        test_sample= self.load_syntax_style_dimension(test_period)
        
        base_path= f"{self.syntax_style_drift_path}/{reference_period}_{test_period}"
        metric_values = {}
        for metric in metrics:
            filename= f"{base_path}_{metric.replace('_drift', '')}.json"
            if ((not os.path.exists(filename)) or 
                (rebase is True)
            ):
                metric_values[metric]= []
            else:
                self.logger.debug(f"Syntactic Style result already exists at {filename}. Skipping calculation.")
                
        metrics= metric_values.keys()

        if len(metrics) == 0:
            return
        
        dimensions= self.load_dimension_names("syntax_style")
        context_distributions= get_syntactic_style_sub_distributions(reference_sample=reference_sample,
                                                                     test_sample=test_sample,
                                                                     dimensions=dimensions)

        

        for sub_distribution in context_distributions:
            ref_dist = np.asarray(sub_distribution["reference_distribution"], dtype=np.float64)
            test_dist = np.asarray(sub_distribution["test_distribution"], dtype=np.float64)
            
            # Filtrar contextos degenerados
            if ref_dist.size < 2 or test_dist.size < 2:
                continue

            if not np.isfinite(ref_dist).all() or not np.isfinite(test_dist).all():
                continue

            if ref_dist.sum() <= 0.0 or test_dist.sum() <= 0.0:
                continue

            sub_distribution_context = sub_distribution["context"]

            sub_distribution_context = re.sub(r"[^A-Za-z0-9_\-+]", "_", sub_distribution_context)
            drift_result = self._calculate_drift(
                reference_sample=ref_dist,
                test_sample=test_dist,
                filename=None,
                metrics=metrics,
                rebase=rebase,
            )

            for metric in metrics:
                if metric not in drift_result:
                    continue
                metric_values[metric].append(drift_result[metric])

        results= {}
        for metric, values in metric_values.items():
            magnitudes = [v["magnitude"] for v in values if v.get("magnitude") is not None and np.isfinite(v["magnitude"])]
            drift={
                "magnitude":        float(np.mean(magnitudes)) if magnitudes else np.nan,
                "magnitude_min":    float(np.min(magnitudes)) if magnitudes else np.nan,
                "magnitude_max":    float(np.max(magnitudes)) if magnitudes else np.nan,
                "magnitude_median": float(np.median(magnitudes)) if magnitudes else np.nan,
                "magnitude_mean":   float(np.mean(magnitudes)) if magnitudes else np.nan,
                "magnitude_std":    float(np.std(magnitudes)) if magnitudes else np.nan,
                # "p_value": float(np.mean(values["p_value"])) if values["p_value"] else np.nan,
            }
            filename= f"{base_path}_{metric.replace('_drift', '')}.json"
            with open(filename, "w") as f:
                json.dump(drift, f)
            results[metric]= drift
        
        return results

        # return self._calculate_drift(reference_sample=reference_sample, test_sample=test_sample, filename=filename, metrics=metrics, rebase=rebase)  

    def calculate_lexical_drift(self, reference_period, test_period, metrics=["js_drift", "kl_drift", "log_drift"], rebase=None):
        self.logger.debug(f"Calculating lexical drift between {reference_period} and {test_period} using metrics: {metrics}")
        reference_sample= self.load_lexical_dimension(reference_period)
        test_sample= self.load_lexical_dimension(test_period)

        filename=f"{self.lexical_drift_path}/{reference_period}_{test_period}"
        return self._calculate_drift(reference_sample=reference_sample, test_sample=test_sample, filename=filename, metrics=metrics, rebase=rebase)

    def calculate_thematic_drift(self, reference_period, test_period, metrics=["js_drift", "kl_drift", "log_drift"], rebase=None):
        self.logger.debug(f"Calculating thematic drift between {reference_period} and {test_period} using metrics: {metrics}")
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
    
    ### Report
    
    
    def _build_report_drift_tables(
        self,
        dimensions=None,
        metrics=None,
        score_key="magnitude",
        fill_diagonal=np.nan,
        sort_periods=True,
    ):
        allowed_dimensions = {
            "semantic": self.load_semantic_drift,
            "syntactic_content": self.load_syntax_content_drift,
            "syntactic_style": self.load_syntax_style_drift,
            "lexical": self.load_lexical_drift,
            "thematic": self.load_thematic_drift,
        }

        if dimensions is None:
            dimensions = list(allowed_dimensions.keys())

        invalid_dimensions = [dim for dim in dimensions if dim not in allowed_dimensions]
        if invalid_dimensions:
            raise ValueError(
                f"Invalid dimensions: {invalid_dimensions}. "
                f"Allowed values are: {list(allowed_dimensions.keys())}"
            )

        if metrics is None:
            metrics = ["mmd", "cos", "kl", "js", "log"]

        periods = self.df["period_id"].dropna().unique().tolist()

        if sort_periods:
            periods = sorted(periods, key=lambda x: int(x))

        tables = {
            dimension: {
                metric: pd.DataFrame(index=periods, columns=periods, dtype=float)
                for metric in metrics
            }
            for dimension in dimensions
        }

        for dimension in dimensions:
            load_method = allowed_dimensions[dimension]

            for metric in metrics:
                self.logger.info(
                    f"Building drift table for dimension '{dimension}' and metric '{metric}'"
                )

                table = tables[dimension][metric]

                for reference_period, test_period in product(periods, periods):
                    if reference_period == test_period:
                        continue

                    try:
                        drift_data = load_method(
                            reference_period=reference_period,
                            test_period=test_period,
                            metric=metric,
                        )

                        if drift_data is None:
                            continue

                        score = drift_data.get(score_key)
                        if score is None:
                            self.logger.warning(
                                f"Score key '{score_key}' not found for "
                                f"{dimension} drift ({reference_period} -> {test_period}, metric={metric})"
                            )
                            continue

                        table.loc[reference_period, test_period] = float(score)

                    except Exception as exc:
                        self.logger.debug(
                            f"Could not load {dimension} drift "
                            f"({reference_period} -> {test_period}, metric={metric}): {exc}"
                        )
                if table.isna().all().all():
                    self.logger.warning(f"Could not load {dimension} drift for metric {metric}")
                    continue

                
                if fill_diagonal is not None:
                    for period in table.index.intersection(table.columns):
                        row = table.loc[period].drop(labels=period, errors="ignore")
                        col = table[period].drop(labels=period, errors="ignore")
                        
                        if not (row.isna().all() and col.isna().all()):
                            table.loc[period, period] = fill_diagonal

                tables[dimension][metric] = table

        cleaned_tables = {}

        for dimension, metric_tables in tables.items():
            cleaned_metric_tables = {}

            for metric, table in metric_tables.items():
                if table.isna().all().all():
                    self.logger.info(
                        f"Removing empty drift table for dimension '{dimension}' and metric '{metric}'"
                    )
                    continue

                table = table.dropna(axis=0, how="all").dropna(axis=1, how="all")
                
                cleaned_metric_tables[metric] = table

            if cleaned_metric_tables:
                cleaned_tables[dimension] = cleaned_metric_tables
            else:
                self.logger.info(
                    f"Removing dimension '{dimension}' because all metric tables are empty"
                )
        self.report_drift_tables = cleaned_tables
        return cleaned_tables


    def report_drift_heatmap(self, dimension, metric, export=False):
        if self.report_drift_tables is None:
            self._build_report_drift_tables()

        titles_dimension = {
            "semantic": "Semantic Drift",
            "syntactic_content": "Syntactic Content Drift",
            "syntactic_style": "Syntactic Style Drift", 
            "lexical": "Lexical Drift",
            "thematic": "Thematic Drift"
        }
        
        tiles_metrics= {
            "mmd": "Maximum Mean Discrepancy, MMD",
            "cos": "Cosine Distance",
            "kl": "Kullback–Leibler Divergence, KL",
            "js": "Jensen–Shannon Divergence, JS",
            "log": "Log-Likelihood",
            "ks": "Kolmogorov–Smirnov Test, KS"
        }
        
        table= self.report_drift_tables[dimension][metric]
        abs_table = table.abs()

        fig, ax = plt.subplots(figsize=(8, 6))

        title= "Absolute Values of " + titles_dimension[dimension]+ " (" + tiles_metrics[metric] + ") Across Periods"
        ax.set_title(title)
        
        sns.heatmap(abs_table, annot=True, cmap="RdYlGn_r", fmt=".3f", ax=ax)
        fig.tight_layout()
        if export:
            filename= f"{self.df_name}_{dimension}_{metric}.svg"
            fig.savefig(f"{self.report_path}/{filename}", format="svg", bbox_inches="tight")
        
        plt.show()
        plt.close(fig)
    
    def report_all_drift_heatmap(self, export=False):
        for dimension, metric_tables in self.report_drift_tables.items():
            for metric in metric_tables.keys():
                self.report_drift_heatmap(dimension=dimension, metric=metric, export=export)
    ###