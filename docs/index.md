# MuSTDrifter

**MuSTDrifter** (**Mu**lti-**S**ource **T**emporal **Drifter**) is a Python framework for unsupervised multidimensional discourse drift detection.

The framework quantifies how discourse evolves over time through complementary forms of distribution shift analysis, combining:

- Covariate shift detection
- Prior probability shift detection
- Multidimensional discourse representations
- Drift estimation and reporting utilities

MuSTDrifter supports discourse analysis across:

- Semantic dimensions
- Lexical dimensions
- Syntactic content
- Syntactic style
- Thematic distributions

## Features

- Multidimensional discourse representations
- Multiple drift metrics
- Parallelized computation
- Permutation-based inference
- Automatic report generation
- Heatmap visualization utilities
- Modular and extensible architecture

## Supported Drift Metrics

### Semantic Drift
- Maximum Mean Discrepancy (MMD)
- Cosine Drift
- Kolmogorov–Smirnov (KS)

### Lexical, Syntactic, and Thematic Drift
- Jensen–Shannon Divergence (JS)
- Kullback–Leibler Divergence (KL)
- Log-Likelihood divergence

## Documentation

- Installation
- Quickstart
- API Reference

## Citation

```bibtex
@article{Guillen-Pacho2026,
  author={Guill{\'e}n-Pacho, Ibai  and Badenes-Olmedo, Carlos and Corcho, Oscar},
  title={An unsupervised multidimensional data drift framework for measuring multilingual and cross-national political discourse shifts},
  journal={Social Network Analysis and Mining},
  year={2026},
  abstract={Political discourse on social networks continuously evolves across electoral cycles and communicative contexts, producing measurable changes in discourse distributions over time. This work introduces an unsupervised framework for detecting and quantifying these changes through multidimensional drift analysis. The approach models discourse evolution across semantic, lexical, thematic, and syntactic dimensions by combining complementary drift detection techniques over text, enabling systematic and reproducible analyses of discourse change. The framework is applied to multilingual political discourse published on party manifestos and Facebook between 2014 and 2023 across multiple electoral cycles and institutional levels in Spain and France. Results show that the proposed framework is able to detect, in a fully unsupervised manner, structural patterns consistent with known dynamics of political communication, including stable distinctions between campaign and routine discourse. At the same time, the framework provides measurable and objective indicators that enable more systematic analyses of discourse evolution across comparable political scenarios.},
  issn={},
  doi={10.1007/s13278-026-01646-9},
  html={https://doi.org/10.1007/s13278-026-01646-9},
  month = {Sep}
}
```
