suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(ggplot2)
  library(ggseg)
})
# =============================
# 1) Get script directory
# =============================
get_script_path <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  idx <- grep(file_arg, cmd_args)
  if (length(idx) > 0) {
    return(normalizePath(sub(file_arg, "", cmd_args[idx])))
  }
  
  if (requireNamespace("rstudioapi", quietly = TRUE)) {
    if (rstudioapi::isAvailable()) {
      return(normalizePath(rstudioapi::getSourceEditorContext()$path))
    }
  }
  
  stop("Cannot determine script path.")
}
script_path <- get_script_path()
script_dir  <- dirname(script_path)
# =============================
# 2) User settings
# =============================
use_clean_atlas <- TRUE
atlas_rds <- file.path(script_dir, "rdata", "dk_nocc.rds")
xlsx_path <- "D:/Codes/cvdproc/cvdproc/data/atlas/desikan_killiany/aparc.xlsx"
sheet_name <- 1
out_png <- "D:/WYJ/Neuroimage/workdir/dk.png"
width <- 9
height <- 4.5
dpi <- 1200
# =============================
# 3) Load atlas
# =============================
if (use_clean_atlas) {
  
  if (!file.exists(atlas_rds)) {
    stop(paste("Clean atlas not found:", atlas_rds))
  }
  
  atlas_use <- readRDS(atlas_rds)
  message("Using cleaned atlas (dk_nocc.rds)")
  
} else {
  
  atlas_use <- dk
  message("Using original ggseg::dk atlas")
  
}
# =============================
# 4) Read xlsx (NO HEADER)
# =============================
df_raw <- read_excel(
  xlsx_path,
  sheet = sheet_name,
  col_names = FALSE
)
if (ncol(df_raw) < 2) {
  stop("The xlsx must contain at least two columns.")
}
names(df_raw)[1:2] <- c("label", "value")
df <- df_raw %>%
  transmute(
    label = as.character(label),
    value = suppressWarnings(as.numeric(value))
  ) %>%
  filter(!is.na(label) & label != "") %>%
  group_by(label) %>%
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop")
# =============================
# 5) Join + plot
# =============================
plot_df <- atlas_use$data %>%
  left_join(df, by = "label")
p <- ggplot(plot_df) +
  geom_brain(
    atlas = atlas_use,
    aes(fill = value),
    color = "grey30",
    size = 0.15
  ) +
  theme_void() +
  scale_fill_viridis_c(option = "D", na.value = "white") +
  labs(fill = "Value")
print(p)
# =============================
# 6) Save
# =============================
ggsave(
  filename = out_png,
  plot = p,
  width = width,
  height = height,
  dpi = dpi
)