"""s3bib - the S3 (Semantic Similarity Score) bibliometric method as a library.

S3 project: https://github.com/AI-Growth-Lab/S3
Paper: https://doi.org/10.1016/j.jbusres.2026.116582
Library code: https://github.com/AI-Growth-Lab/s3-bibliometric
"""
from .pipeline import S3Config, S3Result, run_s3, load_papers, set_seed
from .questions import QUESTIONS, Answer
from .models import EMBEDDING_MODELS, DEFAULT_MODEL
from .scopus import enrich_citations, ScopusAccessError

__all__ = ["S3Config", "S3Result", "run_s3", "load_papers", "set_seed", "QUESTIONS", "Answer",
           "EMBEDDING_MODELS", "DEFAULT_MODEL", "enrich_citations", "ScopusAccessError"]
__version__ = "0.15.0"
