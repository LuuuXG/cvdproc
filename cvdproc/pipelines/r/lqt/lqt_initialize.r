#.libPaths(c("E:/R_packages", .libPaths()))

if (!requireNamespace("rstudioapi", quietly = TRUE)) {
  install.packages("rstudioapi")
}

library("rstudioapi")

cur_dir = dirname(getSourceEditorContext()$path)

extdata_source <- normalizePath(file.path(cur_dir, "..", "..", "..", "data", "lqt", "extdata"),
                                winslash = "/",
                                mustWork = TRUE)

lqt_custom <- normalizePath(file.path(cur_dir, "..", "..", "external", "LQT"),
                            winslash = "/",
                            mustWork = TRUE)

if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools")
}

if (!requireNamespace("LQT", quietly = TRUE)) {
  message("LQT package not found. Installing from local source...")
  devtools::install_local(lqt_custom, dependencies = TRUE)
  #devtools::install_github('jdwor/LQT')
} else {
  message("LQT package is already installed.")
}

extdata_target <- system.file("extdata", package = "LQT")

if (extdata_target == "") {
  stop("LQT package installation not found.")
}

message("Copying data from: ", extdata_source, "\nTo: ", extdata_target)

file.copy(from = list.files(extdata_source, full.names = TRUE),
          to = extdata_target,
          overwrite = TRUE,
          recursive = TRUE)

message("Data copying completed successfully.")