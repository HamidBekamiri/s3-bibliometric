\name{s3_communities}
\alias{s3_communities}
\alias{s3_papers}
\alias{s3_scan}
\alias{s3_search}
\title{Results as data frames}
\description{
\code{s3_communities()} lists the communities with their size, years,
citations, QI and MI. \code{s3_papers()} gives one row per paper with its
community, centrality and outlier flag. \code{s3_scan()} is the threshold scan
(silhouette and SII per threshold). \code{s3_search()} finds the papers closest
in meaning to a query.
}
\usage{
s3_communities(res, min_size = 2L)
s3_papers(res, text = FALSE)
s3_scan(res)
s3_search(res, query, top_k = 10L)
}
\arguments{
  \item{res}{An S3 result from \code{\link{s3_run}} or \code{\link{s3_load}}.}
  \item{min_size}{Smallest community to list.}
  \item{text}{Keep the combined title and abstract column.}
  \item{query}{Search text.}
  \item{top_k}{Number of papers to return.}
}
\value{A data frame.}
\examples{
\dontrun{
res <- s3_run(s3_example())
s3_communities(res)
s3_search(res, "absorptive capacity and innovation")
}
}
