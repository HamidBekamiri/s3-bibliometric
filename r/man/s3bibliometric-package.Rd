\name{s3bibliometric-package}
\alias{s3bibliometric-package}
\alias{s3bibliometric}
\docType{package}
\title{Semantic Similarity Score (S3) for mapping research fields}
\description{
An R interface to the \sQuote{s3-bibliometric} Python library, which implements
the Semantic Similarity Score (S3) method of Bekamiri, Østergaard and Timmermans
(2027). S3 maps the intellectual structure of a research field from the titles
and abstracts of its papers: transformer embeddings, a time-ordered similarity
graph and Leiden communities, with citation indicators (QI, MI) to characterise
the communities.

The R package and the Python library carry the same version number, and the R
package uses exactly that Python version, so both give the same results.

Start with \code{\link{s3_run}}, then \code{\link{s3_communities}},
\code{\link{s3_answer}} and \code{\link{s3_report}}.
}
\references{
Bekamiri, H., Østergaard, C. R., & Timmermans, B. (2027). Semantic similarity
score: A new method for mapping intellectual structures in business research.
\emph{Journal of Business Research}, 219, 116582.
\doi{10.1016/j.jbusres.2026.116582}

S3 project: \url{https://github.com/AI-Growth-Lab/S3}.
Step-by-step guide: \url{https://ai-growth-lab.github.io/S3/}.
}
\keyword{package}
