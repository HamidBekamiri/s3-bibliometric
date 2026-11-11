"""s3bib - the S3 (Semantic Similarity Score) bibliometric method as a library."""
from .pipeline import S3Config, S3Result, run_s3, load_papers

__all__ = ["S3Config", "S3Result", "run_s3", "load_papers"]
__version__ = "0.1.0"
