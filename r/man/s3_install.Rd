\name{s3_install}
\alias{s3_install}
\alias{s3_available}
\alias{s3_version}
\alias{s3_models}
\alias{s3_example}
\title{Python setup, versions, models and example data}
\description{
The package runs the \sQuote{s3-bibliometric} Python library of the same
version. By default \pkg{reticulate} provides it automatically on first use.
\code{s3_install()} installs it into a virtual environment of your own instead.
\code{s3_available()} tells whether the Python library can be loaded,
\code{s3_version()} gives both versions, \code{s3_models()} lists the
embedding models, and \code{s3_example()} is the path of a real example
corpus: 109 OpenAlex records (CC0) on absorptive capacity and work group
diversity.
}
\usage{
s3_install(version = NULL, envname = "r-s3bibliometric", python = NULL, ...)
s3_available()
s3_version()
s3_models()
s3_example()
}
\arguments{
  \item{version}{Python library version; by default the version of this R
    package, or \code{"latest"}.}
  \item{envname}{Name of the virtual environment.}
  \item{python}{Python executable to build the environment from.}
  \item{...}{Passed to \code{reticulate::py_install()}.}
}
\value{\code{s3_install()}: the requirement installed, invisibly.
  \code{s3_available()}: \code{TRUE} or \code{FALSE}. \code{s3_version()}: a list
  with \code{r} and \code{python}. \code{s3_models()}: a data frame.
  \code{s3_example()}: a file path.}
\examples{
s3_example()
\dontrun{
s3_install()
s3_version()
}
}
