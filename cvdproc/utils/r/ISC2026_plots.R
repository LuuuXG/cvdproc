library(ggseg)
library(ggplot2)
library(ggsegJHU)
plot(aseg)

ggplot() +
  geom_brain(atlas = dk, side = "all")

aseg_data <- aseg[["data"]]
ggseg(atlas = jhu)
plot(jhu) +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 7)) +
  guides(fill = guide_legend(ncol = 2))
