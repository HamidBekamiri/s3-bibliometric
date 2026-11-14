\name{s3_report}
\alias{s3_report}
\title{Write the HTML report}
\description{
Writes one HTML report organised for a literature review (brief, theme cards,
research agenda, methods appendix with a methods paragraph, and reading lists),
or every question in order.
}
\usage{
s3_report(res, path = "s3_report.html", level = NULL,
          layout = c("review", "questions"), title = NULL)
}
\arguments{
  \item{res}{An S3 result.}
  \item{path}{Output file.}
  \item{level}{1, 2 or 3; by default the level of the run.}
  \item{layout}{\code{"review"} or \code{"questions"}.}
  \item{title}{Report title.}
}
\value{The path of the report, invisibly.}
\examples{
\dontrun{
res <- s3_run(s3_example())
s3_report(res, "s3_report.html", level = 1)
}
}
