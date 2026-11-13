"""s3bib - the S3 (Semantic Similarity Score) bibliometric method as a library."""
from .pipeline import S3Config, S3Result, run_s3, load_papers, set_seed
from .questions import QUESTIONS, Answer
from .models import EMBEDDING_MODELS, DEFAULT_MODEL
from .scopus import enrich_citations, ScopusAccessError

__all__ = ["S3Config", "S3Result", "run_s3", "load_papers", "set_seed", "QUESTIONS", "Answer",
           "EMBEDDING_MODELS", "DEFAULT_MODEL", "enrich_citations", "ScopusAccessError"]
__version__ = "0.12.0"
