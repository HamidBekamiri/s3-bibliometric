\name{s3_compare_maps}
\alias{s3_compare_maps}
\alias{print.s3_map_choice}
\title{Which map should the analysis use?}
\description{
Compares the text map (the published method) with text plus bibliographic
coupling, co-citation or direct citation at 5\%, 10\%, ... 40\% weight; a
citation counts only for pairs already close in content (the text floor). 20\% of the in-corpus citation links
are hidden while the maps are built; each map is scored on the hidden links
inside its communities, on topic coherence (TCI) and stability, each against
random communities of the same sizes, and on coverage. A rule fixed in advance
recommends a map.
}
\usage{
s3_compare_maps(x, extended = FALSE, model = "mpnet", seed = 42L,
                strict_citations = FALSE, cache_dir = NULL, quiet = FALSE)

\method{print}{s3_map_choice}(x, ...)
}
\arguments{
  \item{x}{An S3 result (uses its citation links, e.g. after
    \code{\link{s3_enrich_citations}}), or a path or data frame as in
    \code{\link{s3_run}}. For \code{print}, a map choice.}
  \item{extended}{Also try the 5-40\% grid without the text floor.}
  \item{model, seed, strict_citations, cache_dir, quiet}{As in
    \code{\link{s3_run}}, when \code{x} is a file or data frame.}
  \item{...}{Ignored.}
}
\value{A list of class \code{s3_map_choice}: \code{table}, \code{recommended},
  \code{reasons}, \code{rule}, \code{citation_links}, \code{hidden_links}.}
\examples{
\dontrun{
ch <- s3_compare_maps(s3_example())
ch$recommended
res <- s3_run(s3_example(), map = "hybrid")
}
}
