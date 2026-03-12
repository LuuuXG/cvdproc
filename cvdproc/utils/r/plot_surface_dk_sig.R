suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(ggplot2)
  library(ggseg)
  library(viridis)
})

plot_dk_from_xlsx <- function(
    xlsx_path,
    out_png = NULL,
    sheet_name = 1,
    use_clean_atlas = TRUE,
    atlas_rds = NULL,
    width = 9,
    height = 4.5,
    dpi = 300,
    background_fill = "white",
    alpha_nonsig = 0.3,
    border_color_nonsig = "grey30",
    border_size_nonsig = 0.5,
    alpha_sig = 1.0,
    border_color_sig = "grey10",
    border_size_sig = 1.0,
    na_fill_color = "white",
    p_threshold = 0.05,
    legend_title = "Value",
    symmetric_colorbar = TRUE,
    midpoint = 0,
    limits = NULL,
    scale_type = c("distiller", "gradient2", "viridis"),
    distiller_palette = "RdBu",
    distiller_direction = -1,
    gradient2_low = "#2166AC",
    gradient2_mid = "white",
    gradient2_high = "#B2182B",
    viridis_option = "D",
    colorbar_barwidth = 2,
    colorbar_barheight = 8,
    colorbar_title_position = "top"
) {
  
  scale_type <- match.arg(scale_type)
  
  if (!file.exists(xlsx_path)) {
    stop(paste("xlsx not found:", xlsx_path))
  }
  
  if (is.null(out_png)) {
    out_png <- sub("\\.xlsx$", ".png", xlsx_path)
  }
  
  if (use_clean_atlas) {
    if (is.null(atlas_rds) || !file.exists(atlas_rds)) {
      stop("use_clean_atlas=TRUE but atlas_rds is missing or not found.")
    }
    atlas_use <- readRDS(atlas_rds)
  } else {
    atlas_use <- dk
  }
  
  df_raw <- read_excel(xlsx_path, sheet = sheet_name, col_names = FALSE)
  
  if (ncol(df_raw) < 3) {
    stop("xlsx must contain at least 3 columns: label, value, p")
  }
  
  names(df_raw)[1:3] <- c("label", "value", "p")
  
  df <- df_raw %>%
    transmute(
      label = as.character(label),
      value = suppressWarnings(as.numeric(value)),
      p = suppressWarnings(as.numeric(p))
    ) %>%
    filter(!is.na(label) & label != "") %>%
    group_by(label) %>%
    summarise(
      value = mean(value, na.rm = TRUE),
      p = min(p, na.rm = TRUE),
      .groups = "drop"
    )
  
  df$value[is.infinite(df$value)] <- NA_real_
  df$p[is.infinite(df$p)] <- NA_real_
  df$sig_flag <- ifelse(!is.na(df$p) & df$p < p_threshold, 1L, 0L)
  
  plot_df <- atlas_use$data %>%
    left_join(df, by = "label")
  
  is_background <- (!is.na(plot_df$side) & plot_df$side == "medial" & is.na(plot_df$label))
  is_sig <- (!is_background) & (!is.na(plot_df$label)) & (plot_df$sig_flag == 1L)
  
  plot_df$bg_alpha <- ifelse(is_background, 1, 0)
  plot_df$nonsig_alpha <- ifelse(!is_background & !is_sig, alpha_nonsig, 0)
  plot_df$sig_alpha <- ifelse(!is_background & is_sig, alpha_sig, 0)
  plot_df$sig_border_size <- ifelse(is_sig, border_size_sig, 0)
  
  if (is.null(limits)) {
    finite_values <- plot_df$value[is.finite(plot_df$value)]
    
    if (length(finite_values) == 0) {
      limits_use <- c(-1, 1)
    } else {
      if (symmetric_colorbar) {
        max_abs <- max(abs(finite_values), na.rm = TRUE)
        if (!is.finite(max_abs) || max_abs == 0) {
          max_abs <- 1
        }
        limits_use <- c(-max_abs, max_abs)
      } else {
        limits_use <- range(finite_values, na.rm = TRUE)
        if (!all(is.finite(limits_use)) || diff(limits_use) == 0) {
          limits_use <- c(-1, 1)
        }
      }
    }
  } else {
    if (length(limits) != 2 || !is.numeric(limits)) {
      stop("limits must be NULL or a numeric vector of length 2")
    }
    limits_use <- limits
  }
  
  fill_scale <- switch(
    scale_type,
    
    "distiller" = scale_fill_distiller(
      palette = distiller_palette,
      direction = distiller_direction,
      limits = limits_use,
      na.value = na_fill_color,
      name = legend_title
    ),
    
    "gradient2" = scale_fill_gradient2(
      low = gradient2_low,
      mid = gradient2_mid,
      high = gradient2_high,
      midpoint = midpoint,
      limits = limits_use,
      na.value = na_fill_color,
      name = legend_title
    ),
    
    "viridis" = scale_fill_viridis_c(
      option = viridis_option,
      limits = limits_use,
      na.value = na_fill_color,
      name = legend_title
    )
  )
  
  title_text <- basename(sub("\\.xlsx$", "", xlsx_path))
  
  p <- ggplot() +
    geom_brain(
      data = plot_df,
      atlas = atlas_use,
      fill = background_fill,
      color = NA,
      aes(alpha = bg_alpha)
    ) +
    geom_brain(
      data = plot_df,
      atlas = atlas_use,
      aes(fill = value, alpha = nonsig_alpha),
      color = border_color_nonsig,
      size = border_size_nonsig
    ) +
    geom_brain(
      data = plot_df,
      atlas = atlas_use,
      aes(fill = value, alpha = sig_alpha, size = sig_border_size),
      color = border_color_sig
    ) +
    fill_scale +
    scale_alpha_identity(guide = "none") +
    scale_size_identity(guide = "none") +
    guides(
      fill = guide_colorbar(
        barwidth = colorbar_barwidth,
        barheight = colorbar_barheight,
        title.position = colorbar_title_position
      )
    ) +
    theme_void() +
    labs(title = title_text)
  
  ggsave(
    filename = out_png,
    plot = p,
    width = width,
    height = height,
    dpi = dpi
  )
  
  invisible(list(
    plot = p,
    out_png = out_png,
    limits = limits_use,
    plot_df = plot_df
  ))
}

plot_dk_from_xlsx(
  xlsx_path = "D:/Codes/cvdproc/cvdproc/utils/r/rdata/aparc.xlsx",
  out_png = "C:/Users/Xiaog/WPSDrive/1136007837/WPS云盘/paper/rssi_glymphatic_analysis/data/analysis/20260303_results/chacho_gmv_ipsi.png",
  use_clean_atlas = TRUE,
  atlas_rds = "D:/Codes/cvdproc/cvdproc/utils/r/rdata/dk_nocc.rds",
  p_threshold = 0.05,
  scale_type = "distiller",
  distiller_palette = "RdBu",
  distiller_direction = -1,
  symmetric_colorbar = TRUE,
  legend_title = "Standardized beta",
  alpha_nonsig = 0.6,
  width = 12,
  height = 6,
  border_size_sig = 0.75
)

batch_plot_dk_dir <- function(
    xlsx_dir,
    out_dir = NULL,
    pattern = "\\.xlsx$",
    recursive = FALSE,
    ...
) {
  
  if (!dir.exists(xlsx_dir)) stop(paste("dir not found:", xlsx_dir))
  
  if (is.null(out_dir)) out_dir <- file.path(xlsx_dir, "png")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  xlsx_files <- list.files(xlsx_dir, pattern = pattern, full.names = TRUE, recursive = recursive)
  if (length(xlsx_files) == 0) stop("No xlsx files found in the directory.")
  
  for (f in xlsx_files) {
    out_png <- file.path(out_dir, paste0(tools::file_path_sans_ext(basename(f)), ".png"))
    plot_dk_from_xlsx(
      xlsx_path = f,
      out_png = out_png,
      ...
    )
  }
  
  message("Done. PNG saved to: ", out_dir)
}

# xlsx_dir <- "C:/Users/Xiaog/WPSDrive/1136007837/WPS云盘/paper/rssi_glymphatic_analysis/data/analysis/20260303_results/glymphatic_roi_atrophy/GMV_contra"
# atlas_rds <- "D:/Codes/cvdproc/cvdproc/utils/r/rdata/dk_nocc.rds"
# batch_plot_dk_dir(
#   xlsx_dir = xlsx_dir,
#   out_dir = file.path(xlsx_dir, "png"),
#   use_clean_atlas = TRUE,
#   atlas_rds = atlas_rds,
#   width = 9,
#   height = 4.5,
#   dpi = 300
# )
