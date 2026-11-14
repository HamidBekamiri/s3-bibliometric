# Tests on real data: 109 OpenAlex records (CC0) with their real all-mpnet-base-v2
# embeddings, served from tests/data so no model download is needed. They run when
# RETICULATE_PYTHON points to a Python with the s3-bibliometric library (as in CI);
# otherwise they are skipped.
if (!nzchar(Sys.getenv("RETICULATE_PYTHON"))) {
  message("Skipping: set RETICULATE_PYTHON to a Python with s3-bibliometric installed.")
  quit(save = "no")
}
library(s3bibliometric)
stopifnot(s3_available())

npz <- normalizePath(file.path("data", "openalex_sample_mpnet.npz"))
reticulate::py_run_string(sprintf("
import numpy as np
from s3bib import pipeline
_z = np.load(r'%s')
_lut = dict(zip(_z['text'].tolist(), _z['emb']))
pipeline.embed_texts = lambda texts, *a, **k: np.array([_lut[t] for t in texts])
", npz))

# R and Python are the same version, so they run the same method
v <- s3_version()
stopifnot(identical(v$r, v$python))

res <- s3_run(s3_example(), quiet = TRUE)
stopifnot(inherits(res, "s3_result"))

# the same result as calling the Python library directly
s3bib <- reticulate::import("s3bib", convert = FALSE)
py <- s3bib$run_s3(s3_example(), progress = reticulate::py_eval("lambda s, p: None"))
py_best <- reticulate::py_to_r(py$best)
cm <- s3_communities(res)
stopifnot(nrow(cm) == nrow(reticulate::py_to_r(py$communities())))
stopifnot(abs(reticulate::py_to_r(res$py$best)$threshold - py_best$threshold) < 1e-12)
stopifnot(sum(cm$n_papers) == py_best$n_papers)

# papers: one row per record, community per paper, no long text column by default
pp <- s3_papers(res)
stopifnot(nrow(pp) == 109, "community" %in% names(pp), !"text" %in% names(pp))
stopifnot(sum(!is.na(pp$community)) == sum(cm$n_papers))

# a data frame gives the same result as the file
df <- utils::read.csv(s3_example(), check.names = FALSE, stringsAsFactors = FALSE,
                      encoding = "UTF-8")
res_df <- s3_run(df, quiet = TRUE)
stopifnot(identical(s3_communities(res_df)$n_papers, cm$n_papers))

# threshold scan: the chosen threshold has the highest SII
sc <- s3_scan(res)
stopifnot(sc$threshold[which.max(sc$sii)] == py_best$threshold)

# questions and answers
q <- s3_questions(res, level = 1)
stopifnot(all(q$level == 1), "Q5" %in% q$id)
a <- s3_answer(res, "q5")
stopifnot(inherits(a, "s3_answer"), a$id == "Q5", length(a$findings) > 0,
          is.data.frame(a$tables[[1]]), nrow(a$tables[[1]]) == nrow(cm))
out <- utils::capture.output(print(a))
stopifnot(any(grepl("Q5", out)))

# the map comparison on this run
ch <- s3_compare_maps(res)
stopifnot(inherits(ch, "s3_map_choice"), "text" %in% ch$table$map,
          ch$recommended %in% ch$table$map, ch$hidden_links > 0)

# report, save, load, export
f <- s3_report(res, tempfile(fileext = ".html"), level = 1)
html <- paste(readLines(f, warn = FALSE), collapse = "\n")
stopifnot(grepl("https://github.com/AI-Growth-Lab/S3", html, fixed = TRUE),
          grepl("10.1016/j.jbusres.2026.116582", html, fixed = TRUE))
d <- s3_save(res, file.path(tempdir(), "s3_run"))
back <- s3_load(d)
stopifnot(identical(s3_communities(back)$n_papers, cm$n_papers))
files <- s3_export(res, file.path(tempdir(), "s3_export"))
stopifnot(all(file.exists(files)))

# models
m <- s3_models()
stopifnot("mpnet" %in% m$key, any(m$default))

cat("All S3 R tests passed.\n")
