\name{s3_answer}
\alias{s3_answer}
\alias{s3_questions}
\alias{print.s3_answer}
\title{Research questions and their answers}
\description{
\code{s3_questions()} lists the ranked research questions for a level of
analysis and whether the data supports each. \code{s3_answer()} answers one
question, e.g. \code{"Q8"} (local and global foundations of each community).
}
\usage{
s3_questions(res, level = NULL, available_only = FALSE)
s3_answer(res, question)

\method{print}{s3_answer}(x, ...)
}
\arguments{
  \item{res}{An S3 result.}
  \item{level}{1, 2 or 3; by default the level the run was made with.}
  \item{available_only}{Only the questions this data can answer.}
  \item{question}{Question id, e.g. \code{"Q5"}.}
  \item{x}{An answer.}
  \item{...}{Ignored.}
}
\value{\code{s3_questions()}: a data frame. \code{s3_answer()}: a list of class
  \code{s3_answer} with \code{id}, \code{question}, \code{findings} (character)
  and \code{tables} (named list of data frames).}
\examples{
\dontrun{
res <- s3_run(s3_example())
s3_questions(res, level = 1)
s3_answer(res, "Q5")$tables[[1]]
}
}
