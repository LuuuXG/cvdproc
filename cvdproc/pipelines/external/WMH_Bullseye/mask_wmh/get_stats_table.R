library(neurobase)
library(oro.nifti)

writeLines('*** Reading Files ***')
arg <- commandArgs(trailingOnly = TRUE)
wmh <- readnii(arg[1])
wmh_mask = wmh[wmh != 0]

writeLines('*** Generating Stats Table ***')
out <- table(wmh_mask)
out <- as.matrix(out)
name <- arg[2]
out = cbind(out, rep(name, nrow(out)))

write.table(out, arg[3], row.names = T)