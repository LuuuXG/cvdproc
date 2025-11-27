import os
import shutil
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, CommandLine, BaseInterfaceInputSpec, BaseInterface
from traits.api import Str, Int, Directory, Bool, Either

# usage: segment_subregions [-h] [--cross CROSS] [--long-base LONG_BASE] [--sd SD] [--suffix SUFFIX]
#                           [--temp-dir TEMP_DIR] [--out-dir OUT_DIR] [--debug] [--threads THREADS]
#                           structure

# Cross-sectional and longitudinal segmentation for the following structures: thalamus, brainstem, hippo-amygdala. To
# segment the thalamic nuclei, for example, in a cross-sectional analysis: segment_subregions thalamus --cross subj
# Similarly, for a longitudinal analysis, run: segment_subregions thalamus --long-base base Timepoints are extracted
# from the `base-tps` file in the `base` subject. Output segmentations and computed structure volumes will be saved to
# the subject's `mri` subdirectory.

# positional arguments:
#   structure             Structure to segment. Options are: thalamus, brainstem, hippo-amygdala.

# optional arguments:
#   -h, --help            show this help message and exit
#   --cross CROSS         Subject to segment in cross-sectional analysis.
#   --long-base LONG_BASE
#                         Base subject for longitudinal analysis. Timepoints are extracted from the base-tps file.
#   --sd SD               Specify subjects directory (will override SUBJECTS_DIR env variable).
#   --suffix SUFFIX       Optional output file suffix.
#   --temp-dir TEMP_DIR   Use alternative temporary directory. This will get deleted unless --debug is enabled.
#   --out-dir OUT_DIR     Use alternative output directory (only for cross-sectional). Default is the subject's `mri`
#                         directory.
#   --debug               Write intermediate debugging outputs.
#   --threads THREADS     Number of threads to use. Defaults to 1.

# unstable in WSL :(
class SegmentSubregionsInputSpec(CommandLineInputSpec):
    structure = Str(mandatory=True, desc="Structure to segment. Options are: thalamus, brainstem, hippo-amygdala.", argstr="%s", position=0)
    cross = Str(mandatory=False, desc="Subject to segment in cross-sectional analysis.", argstr="--cross %s", position=1)
    long_base = Str(mandatory=False, desc="Base subject for longitudinal analysis. Timepoints are extracted from the base-tps file.", argstr="--long-base %s", position=2)
    sd = Str(mandatory=False, desc="Specify subjects directory (will override SUBJECTS_DIR env variable).", argstr="--sd %s", position=3)
    suffix = Str(mandatory=False, desc="Optional output file suffix.", argstr="--suffix %s", position=4) # We don't consider this for output naming
    temp_dir = Str(mandatory=False, desc="Use alternative temporary directory. This will get deleted unless --debug is enabled.", argstr="--temp-dir %s", position=5)
    out_dir = Str(mandatory=False, desc="Use alternative output directory (only for cross-sectional). Default is the subject's `mri` directory.", argstr="--out-dir %s", position=6)
    debug = Bool(mandatory=False, desc="Write intermediate debugging outputs.", argstr="--debug", position=7)
    threads = Int(mandatory=False, desc="Number of threads to use. Defaults to 1.", argstr="--threads %d", position=8)

class SegmentSubregionsOutputSpec(TraitedSpec):
    subject_id = Str(desc="Subject ID")
    subjects_dir = Str(desc="Subjects Directory")

    # if structure == 'hippo-amygdala'
    lh_amygNucVolumes = Either(None, Str, desc="lh.amygNucVolumes-T1.v22.txt")
    rh_amygNucVolumes = Either(None, Str, desc="rh.amygNucVolumes-T1.v22.txt")
    lh_hippoSfVolumes = Either(None, Str, desc="lh.hippoSfVolumes-T1.v22.txt")
    rh_hippoSfVolumes = Either(None, Str, desc="rh.hippoSfVolumes-T1.v22.txt")
    lh_ha_CA_fsspace = Either(None, Str, desc="lh.hippoAmygLabels-T1.v22.CA.FSvoxelSpace.mgz")
    rh_ha_CA_fsspace = Either(None, Str, desc="rh.hippoAmygLabels-T1.v22.CA.FSvoxelSpace.mgz")
    lh_ha_FS60_fsspace = Either(None, Str, desc="lh.hippoAmygLabels-T1.v22.FS60.FSvoxelSpace.mgz")
    rh_ha_FS60_fsspace = Either(None, Str, desc="rh.hippoAmygLabels-T1.v22.FS60.FSvoxelSpace.mgz")
    lh_ha_HBT_fsspace = Either(None, Str, desc="lh.hippoAmygLabels-T1.v22.HBT.FSvoxelSpace.mgz")
    rh_ha_HBT_fsspace = Either(None, Str, desc="rh.hippoAmygLabels-T1.v22.HBT.FSvoxelSpace.mgz")
    lh_ha_fsspace = Either(None, Str, desc="lh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz")
    rh_ha_fsspace = Either(None, Str, desc="rh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz")
    # if structure == 'thalamus
    thalamicNucleiVolumes = Either(None, Str, desc="thalamicNuclei.v13.txt")
    thalamic_fsspace = Either(None, Str, desc="thalamicNuclei.v13.FSvoxelSpace.mgz")
    # if structure == 'brainstem'
    brainstemSsVolumes = Either(None, Str, desc="brainstemSsLabels.v13.txt")
    brainstem_fsspace = Either(None, Str, desc="brainstemSsLabels.v13.FSvoxelSpace.mgz")

class SegmentSubregions(CommandLine):
    input_spec = SegmentSubregionsInputSpec
    output_spec = SegmentSubregionsOutputSpec
    _cmd = "segment_subregions"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        structure = self.inputs.structure
        subjects_dir = self.inputs.sd if self.inputs.sd else os.environ.get("SUBJECTS_DIR", None)
        if not subjects_dir:
            raise ValueError("SUBJECTS_DIR is not set and --sd is not provided.")
        
        subject_id = self.inputs.cross if self.inputs.cross else self.inputs.long_base
        if not subject_id:
            raise ValueError("Either --cross or --long-base must be provided to specify the subject ID.")
        
        mri_dir = os.path.join(subjects_dir, subject_id, "mri")

        outputs["subject_id"] = subject_id
        outputs["subjects_dir"] = subjects_dir

        if structure == 'hippo-amygdala':
            outputs["lh_amygNucVolumes"] = os.path.join(mri_dir, "lh.amygNucVolumes-T1.v22.txt")
            outputs["rh_amygNucVolumes"] = os.path.join(mri_dir, "rh.amygNucVolumes-T1.v22.txt")
            outputs["lh_hippoSfVolumes"] = os.path.join(mri_dir, "lh.hippoSfVolumes-T1.v22.txt")
            outputs["rh_hippoSfVolumes"] = os.path.join(mri_dir, "rh.hippoSfVolumes-T1.v22.txt")
            outputs["lh_ha_CA_fsspace"] = os.path.join(mri_dir, "lh.hippoAmygLabels-T1.v22.CA.FSvoxelSpace.mgz")
            outputs["rh_ha_CA_fsspace"] = os.path.join(mri_dir, "rh.hippoAmygLabels-T1.v22.CA.FSvoxelSpace.mgz")
            outputs["lh_ha_FS60_fsspace"] = os.path.join(mri_dir, "lh.hippoAmygLabels-T1.v22.FS60.FSvoxelSpace.mgz")
            outputs["rh_ha_FS60_fsspace"] = os.path.join(mri_dir, "rh.hippoAmygLabels-T1.v22.FS60.FSvoxelSpace.mgz")
            outputs["lh_ha_HBT_fsspace"] = os.path.join(mri_dir, "lh.hippoAmygLabels-T1.v22.HBT.FSvoxelSpace.mgz")
            outputs["rh_ha_HBT_fsspace"] = os.path.join(mri_dir, "rh.hippoAmygLabels-T1.v22.HBT.FSvoxelSpace.mgz")
            outputs["lh_ha_fsspace"] = os.path.join(mri_dir, "lh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz")
            outputs["rh_ha_fsspace"] = os.path.join(mri_dir, "rh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz")
        elif structure == 'thalamus':
            outputs["thalamicNucleiVolumes"] = os.path.join(mri_dir, "thalamicNuclei.v13.txt")
            outputs["thalamic_fsspace"] = os.path.join(mri_dir, "thalamicNuclei.v13.FSvoxelSpace.mgz")
        elif structure == 'brainstem':
            outputs["brainstemSsVolumes"] = os.path.join(mri_dir, "brainstemSsLabels.v13.txt")
            outputs["brainstem_fsspace"] = os.path.join(mri_dir, "brainstemSsLabels.v13.FSvoxelSpace.mgz")
        
        return outputs

#    segmentHA_T1.sh SUBJECT_ID [SUBJECT_DIR]

#    (the argument [SUBJECT_DIR] is only necessary if the
#     environment variable SUBJECTS_DIR has not been set
#     or if you want to override it)
class SegmentHACrossInputSpec(CommandLineInputSpec):
    subject_id = Str(mandatory=True, desc="Subject ID", argstr="%s", position=0)
    subjects_dir = Str(mandatory=False, desc="Subject Directory", argstr="%s", position=1)

class SegmentHACrossOutputSpec(TraitedSpec):
    subject_id = Str(desc="Subject ID")
    subjects_dir = Str(desc="Subjects Directory")
    lh_amygNucVolumes = Either(None, Str, desc="lh.amygNucVolumes-T1.v22.txt")
    rh_amygNucVolumes = Either(None, Str, desc="rh.amygNucVolumes-T1.v22.txt")
    lh_hippoSfVolumes = Either(None, Str, desc="lh.hippoSfVolumes-T1.v22.txt")
    rh_hippoSfVolumes = Either(None, Str, desc="rh.hippoSfVolumes-T1.v22.txt")
    lh_ha_CA_fsspace = Either(None, Str, desc="lh.hippoAmygLabels-T1.v22.CA.FSvoxelSpace.mgz")
    rh_ha_CA_fsspace = Either(None, Str, desc="rh.hippoAmygLabels-T1.v22.CA.FSvoxelSpace.mgz")
    lh_ha_FS60_fsspace = Either(None, Str, desc="lh.hippoAmygLabels-T1.v22.FS60.FSvoxelSpace.mgz")
    rh_ha_FS60_fsspace = Either(None, Str, desc="rh.hippoAmygLabels-T1.v22.FS60.FSvoxelSpace.mgz")
    lh_ha_HBT_fsspace = Either(None, Str, desc="lh.hippoAmygLabels-T1.v22.HBT.FSvoxelSpace.mgz")
    rh_ha_HBT_fsspace = Either(None, Str, desc="rh.hippoAmygLabels-T1.v22.HBT.FSvoxelSpace.mgz")
    lh_ha_fsspace = Either(None, Str, desc="lh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz")
    rh_ha_fsspace = Either(None, Str, desc="rh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz")

class SegmentHACross(CommandLine):
    input_spec = SegmentHACrossInputSpec
    output_spec = SegmentHACrossOutputSpec
    _cmd = "segmentHA_T1.sh"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        subjects_dir = self.inputs.subjects_dir if self.inputs.subjects_dir else os.environ.get("SUBJECTS_DIR", None)
        if not subjects_dir:
            raise ValueError("SUBJECTS_DIR is not set and subject_dir is not provided.")
        
        subject_id = self.inputs.subject_id
        mri_dir = os.path.join(subjects_dir, subject_id, "mri")

        outputs["subject_id"] = subject_id
        outputs["subjects_dir"] = subjects_dir
        outputs["lh_amygNucVolumes"] = os.path.join(mri_dir, "lh.amygNucVolumes-T1.v22.txt")
        outputs["rh_amygNucVolumes"] = os.path.join(mri_dir, "rh.amygNucVolumes-T1.v22.txt")
        outputs["lh_hippoSfVolumes"] = os.path.join(mri_dir, "lh.hippoSfVolumes-T1.v22.txt")
        outputs["rh_hippoSfVolumes"] = os.path.join(mri_dir, "rh.hippoSfVolumes-T1.v22.txt")
        outputs["lh_ha_CA_fsspace"] = os.path.join(mri_dir, "lh.hippoAmygLabels-T1.v22.CA.FSvoxelSpace.mgz")
        outputs["rh_ha_CA_fsspace"] = os.path.join(mri_dir, "rh.hippoAmygLabels-T1.v22.CA.FSvoxelSpace.mgz")
        outputs["lh_ha_FS60_fsspace"] = os.path.join(mri_dir, "lh.hippoAmygLabels-T1.v22.FS60.FSvoxelSpace.mgz")
        outputs["rh_ha_FS60_fsspace"] = os.path.join(mri_dir, "rh.hippoAmygLabels-T1.v22.FS60.FSvoxelSpace.mgz")
        outputs["lh_ha_HBT_fsspace"] = os.path.join(mri_dir, "lh.hippoAmygLabels-T1.v22.HBT.FSvoxelSpace.mgz")
        outputs["rh_ha_HBT_fsspace"] = os.path.join(mri_dir, "rh.hippoAmygLabels-T1.v22.HBT.FSvoxelSpace.mgz")
        outputs["lh_ha_fsspace"] = os.path.join(mri_dir, "lh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz")
        outputs["rh_ha_fsspace"] = os.path.join(mri_dir, "rh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz")
        return outputs

#    segmentBS.sh SUBJECT_ID [SUBJECT_DIR]

#    (the argument [SUBJECT_DIR] is only necessary if the
#     environment variable SUBJECTS_DIR has not been set
#     or if you want to override it)
class SegmentBSInputSpec(CommandLineInputSpec):
    subject_id = Str(mandatory=True, desc="Subject ID", argstr="%s", position=0)
    subjects_dir = Str(mandatory=False, desc="Subject Directory", argstr="%s", position=1)
class SegmentBSOutputSpec(TraitedSpec):
    subject_id = Str(desc="Subject ID")
    subjects_dir = Str(desc="Subjects Directory")
    brainstemSsVolumes = Either(None, Str, desc="brainstemSsLabels.v13.txt")
    brainstem_fsspace = Either(None, Str, desc="brainstemSsLabels.v13.FSvoxelSpace.mgz")
class SegmentBS(CommandLine):
    input_spec = SegmentBSInputSpec
    output_spec = SegmentBSOutputSpec
    _cmd = "segmentBS.sh"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        subjects_dir = self.inputs.subjects_dir if self.inputs.subjects_dir else os.environ.get("SUBJECTS_DIR", None)
        if not subjects_dir:
            raise ValueError("SUBJECTS_DIR is not set and subject_dir is not provided.")
        
        subject_id = self.inputs.subject_id
        mri_dir = os.path.join(subjects_dir, subject_id, "mri")

        outputs["subject_id"] = subject_id
        outputs["subjects_dir"] = subjects_dir
        outputs["brainstemSsVolumes"] = os.path.join(mri_dir, "brainstemSsLabels.v13.txt")
        outputs["brainstem_fsspace"] = os.path.join(mri_dir, "brainstemSsLabels.v13.FSvoxelSpace.mgz")
        return outputs

#   SUBJECT_ID: FreeSurfer subject name, e.g., bert
#   SUBJECT_DIR: FreeSurfer subjects directory, typically \$SUBJECTS_DIR
#                (the argument [SUBJECT_DIR] is only necessary if the
#                 environment variable SUBJECTS_DIR has not been set
#                 or if you want to override it)
class SegmentThalamicInputSpec(CommandLineInputSpec):
    subject_id = Str(mandatory=True, desc="Subject ID", argstr="%s", position=0)
    subjects_dir = Str(mandatory=False, desc="Subject Directory", argstr="%s", position=1)
class SegmentThalamicOutputSpec(TraitedSpec):
    subject_id = Str(desc="Subject ID")
    subjects_dir = Str(desc="Subjects Directory")
    thalamicNucleiVolumes = Either(None, Str, desc="thalamicNuclei.v13.txt")
    thalamic_fsspace = Either(None, Str, desc="thalamicNuclei.v13.FSvoxelSpace.mgz")
class SegmentThalamic(CommandLine):
    input_spec = SegmentThalamicInputSpec
    output_spec = SegmentThalamicOutputSpec
    _cmd = "segmentThalamicNuclei.sh"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        subjects_dir = self.inputs.subjects_dir if self.inputs.subjects_dir else os.environ.get("SUBJECTS_DIR", None)
        if not subjects_dir:
            raise ValueError("SUBJECTS_DIR is not set and subject_dir is not provided.")
        
        subject_id = self.inputs.subject_id
        mri_dir = os.path.join(subjects_dir, subject_id, "mri")

        outputs["subject_id"] = subject_id
        outputs["subjects_dir"] = subjects_dir
        outputs["thalamicNucleiVolumes"] = os.path.join(mri_dir, "thalamicNuclei.v13.txt")
        outputs["thalamic_fsspace"] = os.path.join(mri_dir, "thalamicNuclei.v13.FSvoxelSpace.mgz")
        return outputs
    
# usage: mri_segment_hypothalamic_subunits [-h] [--s [S [S ...]]] [--sd SD] [--write_posteriors] [--i I] [--o O]
#                                          [--post POST] [--resample RESAMPLE] [--vol VOL] [--crop CROP [CROP ...]]
#                                          [--threads THREADS] [--cpu]

# This module can be run in two modes: a) on FreeSurfer subjects, and b) on any T1-weighted scan(s) of approximatively
# 1mm resolution.

# optional arguments:
#   -h, --help            show this help message and exit
#   --s [S [S ...]]       (required in FS mode) Name of one or several subjects in $SUBJECTS_DIR on which to run
#                         mri_segment_hypothalamic_subunits, assuming recon-all has been run on the specified subjects.
#                         The output segmentations will automatically be saved in each subject's mri folder. If no
#                         argument is given, mri_segment_hypothalamic_subunits will run on all the subjects in
#                         $SUBJECTS_DIR.
#   --sd SD               (FS mode, optional) override current $SUBJECTS_DIR
#   --write_posteriors    (FS mode, optional) save posteriors, default is False
#   --i I                 (required in T1 mode) Image(s) to segment. Can be a path to a single image or to a folder.
#   --o O                 (required in T1 mode) Segmentation output(s). Must be a folder if --i designates a folder.
#   --post POST           (T1 mode, optional) Posteriors output(s). Must be a folder if --i designates a folder.
#   --resample RESAMPLE   (T1 mode, optional) Resampled image(s). Must be a folder if --i designates a folder.
#   --vol VOL             (T1 mode, optional) Output CSV file with volumes for all structures and subjects.
#   --crop CROP [CROP ...]
#                         (both modes, optional) Size of the central patch to analyse (must be divisible by 8). The
#                         whole image is analysed by default.
#   --threads THREADS     (both modes, optional) Number of cores to be used. Default uses 1 core.
#   --cpu                 (both modes, optional) enforce running with CPU rather than GPU.

class HypothalamicSubunitsInputSpec(CommandLineInputSpec):
    s = Either(None, Str(mandatory=False), desc="(required in FS mode) Name of one or several subjects in $SUBJECTS_DIR on which to run mri_segment_hypothalamic_subunits, assuming recon-all has been run on the specified subjects. The output segmentations will automatically be saved in each subject's mri folder. If no argument is given, mri_segment_hypothalamic_subunits will run on all the subjects in $SUBJECTS_DIR.", argstr="--s %s", position=0)
    sd = Str(mandatory=False, desc="(FS mode, optional) override current $SUBJECTS_DIR", argstr="--sd %s", position=1)
    write_posteriors = Bool(mandatory=False, desc="(FS mode, optional) save posteriors, default is False", argstr="--write_posteriors", position=2)
    i = Either(None, Str(mandatory=False), desc="(required in T1 mode) Image(s) to segment. Can be a path to a single image or to a folder.", argstr="--i %s", position=3)
    o = Either(None, Str(mandatory=False), desc="(required in T1 mode) Segmentation output(s). Must be a folder if --i designates a folder.", argstr="--o %s", position=4)
    post = Either(None, Str(mandatory=False), desc="(T1 mode, optional) Posteriors output(s). Must be a folder if --i designates a folder.", argstr="--post %s", position=5)
    resample = Either(None, Str(mandatory=False), desc="(T1 mode, optional) Resampled image(s). Must be a folder if --i designates a folder.", argstr="--resample %s", position=6)
    vol = Either(None, Str(mandatory=False), desc="(T1 mode, optional) Output CSV file with volumes for all structures and subjects.", argstr="--vol %s", position=7)
    crop = Either(None, Int(mandatory=False), desc="(both modes, optional) Size of the central patch to analyse (must be divisible by 8). The whole image is analysed by default.", argstr="--crop %d", position=8)
    threads = Int(mandatory=False, desc="(both modes, optional) Number of cores to be used. Default uses 1 core.", argstr="--threads %d", position=9)
    cpu = Bool(mandatory=False, desc="(both modes, optional) enforce running with CPU rather than GPU.", argstr="--cpu", position=10)

class HypothalamicSubunitsOutputSpec(TraitedSpec):
    hypothalamic_subunits_seg = Either(None, Str, desc="hypothalamic_subunits_seg.v1.mgz")
    hypothalamic_subunits_vol = Either(None, Str, desc="hypothalamic_subunits_volumes.v1.csv")

class HypothalamicSubunits(CommandLine):
    input_spec = HypothalamicSubunitsInputSpec
    output_spec = HypothalamicSubunitsOutputSpec
    _cmd = "mri_segment_hypothalamic_subunits"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        subjects_dir = self.inputs.sd if self.inputs.sd else os.environ.get("SUBJECTS_DIR", None)
        if not subjects_dir and self.inputs.s:
            raise ValueError("SUBJECTS_DIR is not set and --sd is not provided.")
        
        if self.inputs.s:
            subject_id = self.inputs.s
            mri_dir = os.path.join(subjects_dir, subject_id, "mri")
            outputs["hypothalamic_subunits_seg"] = os.path.join(mri_dir, "hypothalamic_subunits_seg.v1.mgz")
            outputs["hypothalamic_subunits_vol"] = os.path.join(mri_dir, "hypothalamic_subunits_volumes.v1.csv")
        elif self.inputs.i and self.inputs.o:
            if os.path.isdir(self.inputs.i):
                base_name = "hypothalamic_subunits_seg.v1.mgz"
                outputs["hypothalamic_subunits_seg"] = os.path.join(self.inputs.o, base_name)
                vol_base_name = "hypothalamic_subunits_volumes.v1.csv"
                outputs["hypothalamic_subunits_vol"] = os.path.join(self.inputs.o, vol_base_name)
            else:
                #TODO: only use FS mode now
                pass
        
        return outputs

if __name__ == "__main__":
    # Example usage
    segmenter = HypothalamicSubunits()
    segmenter.inputs.s = "ses-baseline"
    segmenter.inputs.sd = "/mnt/f/BIDS/demo_BIDS/derivatives/freesurfer/sub-TAOHC0263"
    results = segmenter.run()

    print(results)