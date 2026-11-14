# The R package is an interface to the 's3-bibliometric' Python library. Its
# version matches the Python version it was released with, and on load it asks
# 'reticulate' for exactly that Python version, so R and Python always run the
# same method.

.s3 <- new.env(parent = emptyenv())

python_requirement <- function() {
  paste0("s3-bibliometric==", utils::packageVersion("s3bibliometric"))
}

.onLoad <- function(libname, pkgname) {
  # Users who chose their own Python (RETICULATE_PYTHON, use_python(),
  # use_virtualenv(), or s3_install()) keep it; otherwise reticulate creates a
  # managed environment with the matching Python library on first use.
  own <- nzchar(Sys.getenv("RETICULATE_PYTHON")) ||
    reticulate::virtualenv_exists("r-s3bibliometric")
  if (reticulate::virtualenv_exists("r-s3bibliometric") &&
      !nzchar(Sys.getenv("RETICULATE_PYTHON"))) {
    try(reticulate::use_virtualenv("r-s3bibliometric", required = FALSE), silent = TRUE)
  }
  if (!own) {
    try(reticulate::py_require(python_requirement()), silent = TRUE)
  }
}

# The Python module, imported on first use (convert = FALSE: values are
# converted explicitly, so data frames and lists arrive in a predictable shape).
s3_module <- function(name = "s3bib") {
  key <- name
  if (is.null(.s3[[key]])) {
    if (!reticulate::py_module_available("s3bib")) {
      stop("The S3 Python library is not available. Run s3_install() once and restart R, ",
           "or point reticulate at a Python that has 's3-bibliometric' installed.",
           call. = FALSE)
    }
    .s3[[key]] <- reticulate::import(name, convert = FALSE)
  }
  .s3[[key]]
}
