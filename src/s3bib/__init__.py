"""s3bib - the S3 (Semantic Similarity Score) bibliometric method as a library."""
from .pipeline import S3Config, S3Result, run_s3, load_papers
from .questions import QUESTIONS, Answer

__all__ = ["S3Config", "S3Result", "run_s3", "load_papers", "QUESTIONS", "Answer"]
__version__ = "0.2.0"
