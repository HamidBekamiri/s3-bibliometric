# R interface to the S3 (Semantic Similarity Score) Python library.

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
to_r <- function(x) reticulate::py_to_r(x)

as_df <- function(x) {
  df <- to_r(x)
  if (!is.data.frame(df)) df <- as.data.frame(df, stringsAsFactors = FALSE)
  rownames(df) <- NULL
  df
}

null_or_int <- function(x) if (is.null(x)) NULL else as.integer(x)

check_result <- function(res) {
  if (!inherits(res, "s3_result")) {
    stop("Expected an S3 result from s3_run() or s3_load().", call. = FALSE)
  }
  res$py
}

wrap_result <- function(py) structure(list(py = py), class = "s3_result")

progress_printer <- function(quiet) {
  if (quiet) return(reticulate::py_eval("lambda s, p: None"))
  last <- ""
  function(stage, fraction) {
    val <- function(v) if (inherits(v, "python.builtin.object")) reticulate::py_to_r(v) else v
    stage <- as.character(val(stage))
    fraction <- val(fraction)
    # one line per stage, not one per batch
    key <- sub("[0-9/ ]+$", "", stage)
    if (!identical(key, last)) {
      message(sprintf("%3.0f%%  %s", 100 * as.numeric(fraction), stage))
      last <<- key
    }
    invisible(NULL)
  }
}

config <- function(model = "mpnet", map = "auto", threshold = NULL, start_year = NULL,
                   end_year = NULL, seed = 42L, level = 2L, method = "leiden",
                   strict_citations = FALSE, backbone = FALSE, dedupe = TRUE,
                   cache_dir = NULL, prediction = FALSE) {
  s3_module()$S3Config(
    model_name = model, map_weights = map, threshold = threshold,
    start_year = null_or_int(start_year), end_year = null_or_int(end_year),
    seed = as.integer(seed), level = as.integer(level), community_method = method,
    citation_match = if (isTRUE(strict_citations)) "strict" else "lenient",
    backbone = isTRUE(backbone), dedupe = isTRUE(dedupe),
    cache_dir = if (is.null(cache_dir)) NULL else path.expand(cache_dir),
    prediction = isTRUE(prediction))
}

input_data <- function(data) {
  if (is.data.frame(data)) return(reticulate::r_to_py(data))
  if (is.character(data) && length(data) == 1L) {
    if (!file.exists(data)) stop("File not found: ", data, call. = FALSE)
    return(normalizePath(data))
  }
  stop("`data` must be a path to a Scopus CSV/Excel export or a data frame.", call. = FALSE)
}

# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
s3_install <- function(version = NULL, envname = "r-s3bibliometric", python = NULL, ...) {
  version <- if (is.null(version)) as.character(utils::packageVersion("s3bibliometric")) else version
  pkg <- if (identical(version, "latest")) "s3-bibliometric" else paste0("s3-bibliometric==", version)
  if (!reticulate::virtualenv_exists(envname)) {
    reticulate::virtualenv_create(envname, python = python)
  }
  reticulate::py_install(pkg, envname = envname, method = "virtualenv", pip = TRUE, ...)
  message("Installed ", pkg, " into the Python environment '", envname,
          "'. Restart R, then library(s3bibliometric).")
  invisible(pkg)
}

s3_available <- function() reticulate::py_module_available("s3bib")

s3_version <- function() {
  py <- if (s3_available()) to_r(s3_module()$`__version__`) else NA_character_
  list(r = as.character(utils::packageVersion("s3bibliometric")), python = py)
}

s3_models <- function() {
  pd <- reticulate::import("pandas", convert = FALSE)
  as_df(pd$DataFrame(s3_module("s3bib.models")$model_table()))
}

s3_example <- function() {
  system.file("extdata", "openalex_sample.csv", package = "s3bibliometric", mustWork = TRUE)
}

# ---------------------------------------------------------------------------
# run, save, load
# ---------------------------------------------------------------------------
s3_run <- function(data, model = "mpnet", map = "auto", threshold = NULL,
                   start_year = NULL, end_year = NULL, seed = 42L, level = 2L,
                   method = c("leiden", "louvain"), strict_citations = FALSE,
                   backbone = FALSE, dedupe = TRUE, cache_dir = NULL,
                   prediction = FALSE, quiet = FALSE) {
  method <- match.arg(method)
  cfg <- config(model, map, threshold, start_year, end_year, seed, level, method,
                strict_citations, backbone, dedupe, cache_dir, prediction)
  py <- s3_module()$run_s3(input_data(data), cfg, progress = progress_printer(quiet))
  wrap_result(py)
}

s3_save <- function(res, path) {
  invisible(as.character(reticulate::py_str(check_result(res)$save(path.expand(path)))))
}

s3_load <- function(path) {
  wrap_result(s3_module()$S3Result$load(path.expand(path)))
}

s3_export <- function(res, path) {
  unlist(to_r(check_result(res)$export(path.expand(path))))
}

print.s3_result <- function(x, ...) {
  s <- to_r(x$py$summary())
  map <- to_r(s3_module("s3bib.hybrid")$map_label(x$py$config$map_weights))
  best <- s$best_threshold
  cat("<S3 result>\n")
  cat(sprintf("  %d papers, %s-%s\n", as.integer(s$n_papers), s$year_range[[1]], s$year_range[[2]]))
  cat(sprintf("  map: %s\n", map))
  if (!is.null(best)) {
    cat(sprintf("  threshold %.2f, silhouette %.3f: %d communities, %d outliers\n",
                best$threshold, best$silhouette, as.integer(s$n_communities),
                as.integer(s$n_outliers)))
  } else {
    cat("  no community structure found\n")
  }
  cat(sprintf("  %d in-corpus citation links\n", as.integer(s$n_citation_links)))
  invisible(x)
}

# ---------------------------------------------------------------------------
# results as data frames
# ---------------------------------------------------------------------------
s3_communities <- function(res, min_size = 2L) {
  as_df(check_result(res)$communities(min_size = as.integer(min_size)))
}

s3_papers <- function(res, text = FALSE) {
  df <- as_df(check_result(res)$papers)
  if (!isTRUE(text)) df$text <- NULL
  df
}

s3_scan <- function(res) {
  pd <- reticulate::import("pandas", convert = FALSE)
  as_df(pd$DataFrame(check_result(res)$scan))
}

s3_search <- function(res, query, top_k = 10L) {
  df <- as_df(check_result(res)$search(query, top_k = as.integer(top_k)))
  df$text <- NULL
  df
}

# ---------------------------------------------------------------------------
# questions and answers
# ---------------------------------------------------------------------------
s3_questions <- function(res, level = NULL, available_only = FALSE) {
  as_df(check_result(res)$questions(available_only = isTRUE(available_only),
                                    level = null_or_int(level)))
}

s3_answer <- function(res, question) {
  a <- check_result(res)$answer(toupper(question))
  tables <- to_r(a$tables)
  tables <- lapply(tables, function(t) { rownames(t) <- NULL; t })
  structure(list(id = to_r(a$qid), question = to_r(a$question),
                 findings = as.character(unlist(to_r(a$findings))),
                 tables = tables),
            class = "s3_answer")
}

print.s3_answer <- function(x, ...) {
  cat(x$id, " ", x$question, "\n", sep = "")
  for (f in x$findings) cat(strwrap(f, indent = 2, exdent = 4, prefix = ""), sep = "\n")
  if (length(x$tables)) cat("\nTables:", paste0("$tables[[\"", names(x$tables), "\"]]"), sep = "\n  ")
  invisible(x)
}

s3_report <- function(res, path = "s3_report.html", level = NULL,
                      layout = c("review", "questions"), title = NULL) {
  layout <- match.arg(layout)
  out <- check_result(res)$report(path.expand(path), title = title,
                                  level = null_or_int(level), layout = layout)
  invisible(as.character(reticulate::py_str(out)))
}

# ---------------------------------------------------------------------------
# map comparison and Scopus references
# ---------------------------------------------------------------------------
s3_compare_maps <- function(x, extended = FALSE, model = "mpnet", seed = 42L,
                            strict_citations = FALSE, cache_dir = NULL, quiet = FALSE) {
  if (inherits(x, "s3_result")) {
    ch <- x$py$compare_maps(extended = isTRUE(extended), refresh = TRUE)
  } else {
    cfg <- config(model = model, seed = seed, strict_citations = strict_citations,
                  cache_dir = cache_dir)
    ch <- s3_module("s3bib.mapchoice")$compare_maps(input_data(x), cfg,
                                                    extended = isTRUE(extended),
                                                    progress = progress_printer(quiet))
  }
  get <- function(k) to_r(reticulate::py_get_item(ch, k))
  t <- reticulate::py_get_item(ch, "table")
  t <- as_df(t$drop(columns = c("weights", "significant ties"), errors = "ignore"))
  structure(list(table = t, recommended = get("recommended"),
                 reasons = as.character(unlist(get("reasons"))), rule = get("rule"),
                 citation_links = get("n_links"), hidden_links = get("n_heldout")),
            class = "s3_map_choice")
}

print.s3_map_choice <- function(x, ...) {
  cols <- intersect(c("map", "held-out citations gain", "TCI gain over random",
                      "stability (ARI)", "coverage", "communities"), names(x$table))
  print(x$table[, cols, drop = FALSE], digits = 3, row.names = FALSE)
  cat("\nRecommended map:", x$recommended, "\n")
  for (r in x$reasons) cat(strwrap(r, indent = 2, exdent = 4), sep = "\n")
  invisible(x)
}

s3_enrich_citations <- function(res, rebuild = TRUE, quiet = FALSE) {
  key <- Sys.getenv("SCOPUS_API_KEY")
  if (!nzchar(key)) {
    stop("Set the SCOPUS_API_KEY environment variable first (e.g. in ~/.Renviron); ",
         "never type the key into a script you share.", call. = FALSE)
  }
  token <- Sys.getenv("SCOPUS_INST_TOKEN")
  to_r(s3_module("s3bib.scopus")$enrich_citations(
    check_result(res), api_key = key, inst_token = if (nzchar(token)) token else NULL,
    rebuild = isTRUE(rebuild), progress = progress_printer(quiet)))
}
