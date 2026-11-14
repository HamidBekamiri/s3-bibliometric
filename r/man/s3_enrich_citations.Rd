\name{s3_enrich_citations}
\alias{s3_enrich_citations}
\title{Complete reference lists from the Scopus API}
\description{
Fetches every paper's reference list from the Scopus API by its EID: in-corpus
citations become exact, and bibliographic coupling uses cited Scopus IDs
instead of the exported reference text. If the map uses citations, it is
rebuilt on the exact data.
}
\usage{
s3_enrich_citations(res, rebuild = TRUE, quiet = FALSE)
}
\arguments{
  \item{res}{An S3 result.}
  \item{rebuild}{Rebuild the map when it uses citation relations.}
  \item{quiet}{Hide progress messages.}
}
\details{Needs a Scopus API key in the \env{SCOPUS_API_KEY} environment
  variable (for example in \file{~/.Renviron}) and your institution's Scopus
  access; an institutional token can be set in \env{SCOPUS_INST_TOKEN}.}
\value{A list with the number of papers queried, failures and citation links
  before and after.}
\examples{
\dontrun{
res <- s3_run("scopus.csv")
s3_enrich_citations(res)
}
}
