#.libPaths(c("E:/R_packages", .libPaths()))
library("LQT")

Sys.setenv(QT_PLUGIN_PATH = "")
Sys.setenv(QT_QPA_PLATFORM_PLUGIN_PATH = "")
Sys.setenv(QT_DEBUG_PLUGINS = "0")

patient_id <- '/this/is/for/nipype/patient_id'
lesion_file <- '/this/is/for/nipype/source_lesion_file'
output_dir <- '/this/is/for/nipype/output_dir'
parcel_path <- '/this/is/for/nipype/parcel_path'

dsi_path <- '/this/is/for/nipype/dsi_path'

#print(dsi_path)

# 1. Set up configuration for LQT
cfg = create_cfg_object(pat_ids=patient_id,
                        lesion_paths=lesion_file,
                        parcel_path=parcel_path,
                        out_path=output_dir,
                        dsi_path=dsi_path)

# 2. Run the LQT analysis
# make output directory if it doesn't exist
if (!dir.exists(cfg$out_path)) {
  dir.create(cfg$out_path, recursive = TRUE)
}

# Get parcel damage for patients
get_parcel_damage(cfg, cores=4)
# Get tract SDC for patients
get_tract_discon(cfg, cores=4)
# Get parcel SDC and SSPL measures for patients
get_parcel_cons(cfg, cores=4)
