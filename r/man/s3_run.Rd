\name{s3_run}
\alias{s3_run}
\alias{print.s3_result}
\title{Run the S3 analysis}
\description{
Embeds each paper's title and abstract, builds the time-ordered S3 similarity
graph, chooses the threshold by the Sensitivity Inclusion Index (SII) and finds
the communities with Leiden, as in the published method.
}
\usage{
s3_run(data, model = "mpnet", map = "auto", threshold = NULL,
       start_year = NULL, end_year = NULL, seed = 42L, level = 2L,
       method = c("leiden", "louvain"), strict_citations = FALSE,
       backbone = FALSE, dedupe = TRUE, cache_dir = NULL,
       prediction = FALSE, quiet = FALSE)

\method{print}{s3_result}(x, ...)
}
\arguments{
  \item{data}{Path to a Scopus CSV or Excel export, or a data frame with at
    least \code{Title} and \code{Abstract} columns. \code{Year}, \code{Cited by},
    \code{Authors}, \code{Affiliations}, \code{Author Keywords},
    \code{References} and \code{EID} are used when present.}
  \item{model}{Embedding model: \code{"mpnet"} (default, the published
    method), another key from \code{\link{s3_models}}, or any
    sentence-transformers model name.}
  \item{map}{Which relations build the map. \code{"auto"} (default) compares
    the text map with text plus each citation relation at 5-40\% weight and
    uses a hybrid only when it raises both silhouette and QI (see
    \code{\link{s3_compare_maps}}). \code{"text"} is the published method
    only; \code{"text+coupling"}, \code{"text+cocitation"},
    \code{"text+direct"} or weights such as \code{"s3=0.8,coupling=0.2"} fix a
    hybrid.}
  \item{threshold}{Fix the similarity threshold (e.g. \code{0.78}) instead of
    choosing it by SII.}
  \item{start_year, end_year}{Optional year range.}
  \item{seed}{Random seed; the same seed gives the same result.}
  \item{level}{Level of analysis for questions and reports: 1 the published
    method, 2 adds two-mode networks and the knowledge graph, 3 adds
    prediction.}
  \item{method}{\code{"leiden"} (published method) or \code{"louvain"} (the S3
    web app).}
  \item{strict_citations}{Short titles only count as cited when year and first
    author also match.}
  \item{backbone}{Keep only significant coupling and co-citation ties.}
  \item{dedupe}{Remove duplicate records before the analysis.}
  \item{cache_dir}{Folder to cache embeddings, so later runs are fast.}
  \item{prediction}{Include the prediction questions (Q24-Q34).}
  \item{quiet}{Hide progress messages.}
  \item{x}{An S3 result.}
  \item{...}{Ignored.}
}
\value{An object of class \code{s3_result}. Read it with
  \code{\link{s3_communities}}, \code{\link{s3_papers}},
  \code{\link{s3_answer}} and \code{\link{s3_report}}.}
\details{The first run downloads the embedding model (about 420 MB for mpnet).}
\examples{
\dontrun{
res <- s3_run(s3_example())
res
s3_communities(res)
}
}
