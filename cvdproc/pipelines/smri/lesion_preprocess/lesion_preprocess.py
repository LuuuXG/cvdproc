import os
from nipype.interfaces.base import (CommandLineInputSpec, CommandLine, traits, TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File)
from traits.api import Directory, Str, List, Bool, Enum, Int, Float, TraitError, Either
from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import isdefined


from nipype.interfaces.base import (
    Undefined,
)

from cvdproc.config.paths import get_package_path

# ============================================
# Native-space lesion filling using SynthMorph
#
# Final output:
#   A single full-head lesion-filled T1 image at the path
#   specified by -o (no separate brain-only output).
#
# Requirements:
#   - FSL (fslreorient2std, fslswapdim, fslmaths)
#   - FreeSurfer 7.3+ (mri_synthstrip, mri_synthmorph)
#
# Usage:
#   lesion_fill_synthmorph_native.sh \
#       -t T1.nii.gz \
#       -m lesion_mask.nii.gz \
#       -o filled_T1_out.nii.gz \
#       [-j filled_T1_out.json] \
#       [-s smooth_sigma] \
#       [-d dilate_iters]
#
# ============================================

class LeftRightLesionFillInputSpec(CommandLineInputSpec):
    t1w_file = File(exists=True, desc="Input T1-weighted image", mandatory=True, argstr='-t %s')
    lesion_mask = File(exists=True, desc="Input lesion mask image", mandatory=True, argstr='-m %s')
    output_file = Str(desc="Output lesion-filled T1 image", mandatory=True, argstr='-o %s')
    output_json = Str(desc="Output JSON metadata file", argstr='-j %s')
    smooth_sigma = Float(desc="Gaussian smoothing sigma for lesion mask", argstr='-s %f', default=1)
    dilate_iters = Int(desc="Number of dilation iterations for lesion mask", argstr='-d %d', default=1)

class LeftRightLesionFillOutputSpec(TraitedSpec):
    output_file = File(desc="Output lesion-filled T1 image")
    output_json = Either(File, None, desc="Output JSON metadata file")

class LeftRightLesionFill(CommandLine):
    """
    Lesion filling using left-right hemisphere mirroring via SynthMorph.
    """
    _cmd = get_package_path('pipelines', 'bash', 'lesion_fill', 'lesion_fill_synthmorph.sh')
    input_spec = LeftRightLesionFillInputSpec
    output_spec = LeftRightLesionFillOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.output_file):
            outputs['output_file'] = os.path.abspath(self.inputs.output_file)
        outputs['output_json'] = None
        if isdefined(self.inputs.output_json):
            outputs['output_json'] = os.path.abspath(self.inputs.output_json) if self.inputs.output_json is not None else None
        return outputs

class SymmetricMniLesionFillInputSpec(CommandLineInputSpec):
    t1w_file = File(
        exists=True,
        desc="Input T1-weighted image",
        mandatory=True,
        argstr="--t1 %s",
    )
    lesion_mask = File(
        exists=True,
        desc="Input lesion mask image in native T1 space",
        mandatory=True,
        argstr="--lesion-mask %s",
    )
    mni_template = File(
        exists=True,
        desc="Symmetric MNI T1 template",
        mandatory=True,
        argstr="--mni-template %s",
    )

    warp_fwd = File(
        exists=False,
        desc="Output forward warp (T1 -> MNI)",
        mandatory=True,
        argstr="--warp-fwd %s",
    )
    warp_inv = File(
        exists=False,
        desc="Output inverse warp (MNI -> T1)",
        mandatory=True,
        argstr="--warp-inv %s",
    )

    contra_mask = File(
        exists=False,
        desc="Output contralateral lesion mask in native T1 space",
        mandatory=True,
        argstr="--contra-mask %s",
    )

    contra_only = Bool(
        False,
        usedefault=True,
        desc="Only generate contralateral mask and skip filled T1 and JSON",
        argstr="--contra-only",
    )

    t1_mni = File(
        exists=False,
        desc="Output T1 registered to MNI",
        mandatory=False,
        argstr="--t1-mni %s",
    )
    filled_t1 = File(
        exists=False,
        desc="Output lesion-filled T1 in native space",
        mandatory=False,
        argstr="--filled-t1 %s",
    )
    bids_dir = Directory(
        exists=True,
        desc="BIDS dataset root directory (for JSON Sources)",
        mandatory=False,
        argstr="--bids-dir %s",
    )


class SymmetricMniLesionFillOutputSpec(TraitedSpec):
    warp_fwd = File(desc="Output forward warp (T1 -> MNI)", exists=False)
    warp_inv = File(desc="Output inverse warp (MNI -> T1)", exists=False)
    t1_mni_sym = File(desc="Output T1 registered to MNI", exists=False)
    filled_t1 = File(desc="Output lesion-filled T1 in native space", exists=False)
    contra_mask = File(desc="Output contralateral lesion mask in native space", exists=False)


class SymmetricMniLesionFill(CommandLine):
    """
    Lesion filling using symmetric MNI-based registration via SynthMorph.
    """
    _cmd = get_package_path("pipelines", "bash", "lesion_fill", "lesion_fill_symmMNI_synthmorph.sh")
    input_spec = SymmetricMniLesionFillInputSpec
    output_spec = SymmetricMniLesionFillOutputSpec

    def _validate_required_inputs(self):
        """
        Enforce conditional requirements:
        - If contra_only is False: t1_mni, filled_t1, bids_dir must be provided.
        - If contra_only is True: those are optional.
        """
        if not getattr(self.inputs, "contra_only", False):
            missing = []
            if not isdefined(self.inputs.t1_mni):
                missing.append("t1_mni")
            if not isdefined(self.inputs.filled_t1):
                missing.append("filled_t1")
            if not isdefined(self.inputs.bids_dir):
                missing.append("bids_dir")
            if missing:
                raise ValueError(
                    "Missing required inputs for default mode (contra_only=False): "
                    + ", ".join(missing)
                )

    def _parse_inputs(self, skip=None):
        self._validate_required_inputs()
        return super()._parse_inputs(skip=skip)

    def _list_outputs(self):
        outputs = self.output_spec().get()

        outputs["warp_fwd"] = os.path.abspath(self.inputs.warp_fwd)
        outputs["warp_inv"] = os.path.abspath(self.inputs.warp_inv)
        outputs["contra_mask"] = os.path.abspath(self.inputs.contra_mask)

        if getattr(self.inputs, "contra_only", False):
            outputs["t1_mni_sym"] = Undefined
            outputs["filled_t1"] = Undefined
        else:
            outputs["t1_mni_sym"] = os.path.abspath(self.inputs.t1_mni)
            outputs["filled_t1"] = os.path.abspath(self.inputs.filled_t1)

        return outputs


def isdefined(x):
    return x is not Undefined and x is not None and not (hasattr(x, "defined") and not x.defined)
    
# ======================================================
# Fastsurfer Lesion Inpainting Tool (LIT)
# Usage: run_lit_containerized_custom.sh \
#   --input_image <input_t1w_volume> \
#   --mask_image <lesion_mask_volume> \
#   --output_directory <output_directory> \
#   --lit_data_dir <lit_data_dir> \
#   [--output_image <output_image>] \
#   [OPTIONS]

# This script runs LIT (Lesion Inpainting Tool) inside Docker and creates:
#   (i)  an inpainted T1w image using a lesion mask
#   (ii) (optional) whole brain segmentation and cortical surface reconstruction using FastSurferVINN

# REQUIRED:
#   -i, --input_image <input_image>
#       Path to the input T1w volume
#   -m, --mask_image <mask_image>
#       Path to the lesion mask volume
#   -o, --output_directory <output_directory>
#       Path to the output directory
#   --lit_data_dir <lit_data_dir>
#       Directory containing LIT data (must include weights/)

# OPTIONAL:
#   --output_image <output_image>
#       Final output image path. If set, the inpainted image
#       (inpainting_volumes/inpainting_result.nii.gz) will be copied to this path.
#       After copying, inpainting_images/ and inpainting_volumes/ under output_directory
#       will be removed.
#   --gpus <gpus>
#       GPUs to use. Default: all
#   --fastsurfer
#       Run FastSurferVINN (requires FreeSurfer license)
#   --fs_license <fs_license>
#       Path to FreeSurfer license file (license.txt)
#   -h, --help
#       Print this message and exit
#   --version
#       Print the version number and exit

# Examples:
#   ./run_lit_containerized_custom.sh \
#     -i t1w.nii.gz \
#     -m lesion.nii.gz \
#     -o ./output \
#     --lit_data_dir /mnt/e/codes/cvdproc/cvdproc/data/lit \
#     --dilate 2

#   ./run_lit_containerized_custom.sh \
#     -i t1w.nii.gz \
#     -m lesion.nii.gz \
#     -o ./output \
#     --lit_data_dir /mnt/e/codes/cvdproc/cvdproc/data/lit \
#     --output_image /path/to/final_inpainted.nii.gz \
#     --dilate 2
# ======================================================
class LITInputSpec(CommandLineInputSpec):
    input_image = File(exists=True, desc="Input T1-weighted image", mandatory=True, argstr='--input_image %s')
    mask_image = File(exists=True, desc="Input lesion mask image", mandatory=True, argstr='--mask_image %s')
    output_directory = Str(desc="Output directory", mandatory=True, argstr='--output_directory %s')
    lit_data_dir = Str(desc="LIT data directory", mandatory=True, argstr='--lit_data_dir %s')
    output_image = Str(desc="Final output inpainted image", argstr='--output_image %s')
    gpus = Str(desc="GPUs to use", argstr='--gpus %s')
    fastsurfer = Bool(desc="Run FastSurferVINN", argstr='--fastsurfer')
    fs_license = File(exists=True, desc="FreeSurfer license file", argstr='--fs_license %s')

class LITOutputSpec(TraitedSpec):
    output_image = File(desc="Final output inpainted image")

class LIT(CommandLine):
    _cmd = 'bash ' + get_package_path('pipelines', 'external', 'LIT', 'run_lit_containerized_custom.sh')
    input_spec = LITInputSpec
    output_spec = LITOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.output_image):
            outputs['output_image'] = os.path.abspath(self.inputs.output_image)
        else:
            outputs['output_image'] = None
        return outputs

# ======================================================
# Write BIDS json sidecar for lesion-filled T1 image
# ======================================================
class LesionFillBidsJsonInputSpec(BaseInterfaceInputSpec):
    json_out = Str(desc="Output JSON file path", mandatory=True)
    bids_dir = Str(desc="BIDS dataset root directory", mandatory=True)
    t1w_file = File(exists=True, desc="Input T1-weighted image", mandatory=True)
    lesion_mask = File(exists=True, desc="Input lesion mask image", mandatory=True)

class LesionFillBidsJsonOutputSpec(TraitedSpec):
    json_out = File(desc="Output JSON file path")

class LesionFillBidsJson(BaseInterface):
    """
    Write BIDS JSON sidecar for lesion-filled T1 image.
    """

    input_spec = LesionFillBidsJsonInputSpec
    output_spec = LesionFillBidsJsonOutputSpec

    def _run_interface(self, runtime):
        # Create BIDS URIs for sources
        # {
        # "Modality": "MR"
        # ,"Sources": [
        #     "bids::sub-SSI0140/ses-F1/anat/sub-SSI0140_ses-F1_acq-highres_T1w.nii.gz"
        #     ,"bids::derivatives/lesion_mask/sub-SSI0140/ses-F1/sub-SSI0140_ses-F1_space-T1w_desc-RSSI_mask.nii.gz"
        # ]
        # }
        import json
        content = {}
        content['Modality'] = 'MR'
        sources = []
        # Create BIDS URI (replace BIDS root path with 'bids::')
        bids_dir = os.path.abspath(self.inputs.bids_dir)
        t1w_path = os.path.abspath(self.inputs.t1w_file)
        lesion_mask_path = os.path.abspath(self.inputs.lesion_mask) 
        t1w_bids_uri = 'bids::' + os.path.relpath(t1w_path, bids_dir).replace('\\', '/')
        lesion_mask_bids_uri = 'bids::' + os.path.relpath(lesion_mask_path, bids_dir).replace('\\', '/')
        sources.append(t1w_bids_uri)
        sources.append(lesion_mask_bids_uri)
        content['Sources'] = sources
        # Write to JSON
        with open(self.inputs.json_out, 'w') as f:
            json.dump(content, f, indent=4)
        self._results = {}
        self._results['json_out'] = os.path.abspath(self.inputs.json_out)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['json_out'] = self._results['json_out']
        return outputs


# ======================================================
# Lesion size analysis (ensure only one cluster)
# volume: voxel * size
# largest diameter (axial): in-plane max distance between any two points in the lesion mask
# largest diameter (3D): max distance between any two points in the lesion mask in 3D space
# ======================================================
class LesionSizeAnalysisInputSpec(BaseInterfaceInputSpec):
    lesion_mask = File(exists=True, desc="Input lesion mask image", mandatory=True)
    out_csv = Str(desc="Output CSV file for lesion size metrics", mandatory=False)

class LesionSizeAnalysisOutputSpec(TraitedSpec):
    volume_mm3 = Float(desc="Lesion volume in cubic millimeters")
    largest_diameter_axial_mm = Float(desc="Largest diameter in axial plane in millimeters")
    largest_diameter_3d_mm = Float(desc="Largest diameter in 3D space in millimeters")
    out_csv = File(desc="Output CSV file for lesion size metrics")

class LesionSizeAnalysis(BaseInterface):
    """
    Analyze lesion size metrics.

    Steps:
      1) Load lesion mask.
      2) Binarize (threshold > 0.5).
      3) Keep only the largest connected component (3D connectivity).
      4) Compute:
         - Volume (mm^3) = voxel_count * voxel_volume
         - Largest axial diameter (mm): max in-plane distance within any single axial slice (max over z)
         - Largest 3D diameter (mm): max Euclidean distance across all lesion voxels in 3D space
    """

    input_spec = LesionSizeAnalysisInputSpec
    output_spec = LesionSizeAnalysisOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        import numpy as np
        from scipy.spatial.distance import pdist
        from scipy import ndimage

        lesion_img = nib.load(self.inputs.lesion_mask)
        lesion_data = lesion_img.get_fdata()
        voxel_dims = lesion_img.header.get_zooms()[:3]  # (x,y,z) in mm

        # Binarize
        mask = lesion_data > 0.5

        # If empty mask, return zeros
        if not np.any(mask):
            self._results["volume_mm3"] = 0.0
            self._results["largest_diameter_axial_mm"] = 0.0
            self._results["largest_diameter_3d_mm"] = 0.0
            return runtime

        # Keep only the largest connected component in 3D
        # Use 26-connectivity by default (full connectivity in 3D)
        structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
        labeled, num = ndimage.label(mask, structure=structure)

        if num > 1:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0  # background
            largest_label = int(np.argmax(counts))
            mask = labeled == largest_label

        # Compute volume
        voxel_volume = float(np.prod(voxel_dims))
        num_voxels = int(np.count_nonzero(mask))
        volume_mm3 = float(num_voxels * voxel_volume)

        # Get voxel coordinates (i,j,k) for lesion voxels
        coords = np.column_stack(np.where(mask))  # (N,3)

        # Largest axial diameter: max over all z slices
        largest_diameter_axial_mm = 0.0
        if coords.shape[0] > 1:
            x_sp, y_sp, _ = voxel_dims
            unique_z = np.unique(coords[:, 2])
            for z in unique_z:
                slice_xy = coords[coords[:, 2] == z][:, :2]  # (M,2)
                if slice_xy.shape[0] > 1:
                    slice_xy_mm = slice_xy * np.array([x_sp, y_sp], dtype=float)
                    d = pdist(slice_xy_mm, metric="euclidean")
                    if d.size > 0:
                        m = float(d.max())
                        if m > largest_diameter_axial_mm:
                            largest_diameter_axial_mm = m

        # Largest 3D diameter
        largest_diameter_3d_mm = 0.0
        if coords.shape[0] > 1:
            coords_mm = coords * np.array(voxel_dims, dtype=float)
            d3 = pdist(coords_mm, metric="euclidean")
            if d3.size > 0:
                largest_diameter_3d_mm = float(d3.max())

        self._results = {}
        self._results["volume_mm3"] = volume_mm3
        self._results["largest_diameter_axial_mm"] = largest_diameter_axial_mm
        self._results["largest_diameter_3d_mm"] = largest_diameter_3d_mm

        # save to CSV (one row)
        if isdefined(self.inputs.out_csv) and self.inputs.out_csv:
            import csv
            with open(self.inputs.out_csv, mode='w', newline='') as csvfile:
                fieldnames = ['volume_mm3', 'largest_diameter_axial_mm', 'largest_diameter_3d_mm']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({
                    'volume_mm3': volume_mm3,
                    'largest_diameter_axial_mm': largest_diameter_axial_mm,
                    'largest_diameter_3d_mm': largest_diameter_3d_mm
                })
            self._results['out_csv'] = os.path.abspath(self.inputs.out_csv)
        else:
            self._results['out_csv'] = None

        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()

        outputs["volume_mm3"] = self._results["volume_mm3"]
        outputs["largest_diameter_axial_mm"] = self._results["largest_diameter_axial_mm"]
        outputs["largest_diameter_3d_mm"] = self._results["largest_diameter_3d_mm"]
        outputs["out_csv"] = self._results["out_csv"]

        return outputs