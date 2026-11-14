\name{s3_save}
\alias{s3_save}
\alias{s3_load}
\alias{s3_export}
\title{Save, load and export a run}
\description{
\code{s3_save()} stores a run in a folder that \code{s3_load()} reads back,
also from Python (\code{S3Result.load}). \code{s3_export()} writes CSV files
of papers, communities and the threshold scan.
}
\usage{
s3_save(res, path)
s3_load(path)
s3_export(res, path)
}
\arguments{
  \item{res}{An S3 result.}
  \item{path}{Folder.}
}
\value{\code{s3_save()}: the folder, invisibly. \code{s3_load()}: an S3 result.
  \code{s3_export()}: the paths of the files written.}
\examples{
\dontrun{
res <- s3_run(s3_example())
s3_save(res, "my_run")
res <- s3_load("my_run")
}
}
