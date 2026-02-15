import os
import subprocess
import shutil
import nibabel as nib
import numpy as np
import pandas as pd
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool, Int, Str, Float, List, Dict

from cvdproc.config.paths import get_package_path

####################################
# Generate b0_all and acqparam.txt #
####################################
class B0AllAndAcqparamInputSpec(BaseInterfaceInputSpec):
    dwi_img = File(desc="Path to the DWI image. First volume is b0 image")
    dwi_bval = File(desc="Path to the bval file")
    phase_encoding_number = traits.Str(desc="Phase encoding number")
    total_readout_time = traits.Str(desc="Total readout time")

    reverse_dwi_img = File(desc="Path to reverse phase-encoded DWI image")
    reverse_dwi_bval = File(desc="Path to bval file for reverse DWI (optional)")
    reverse_phase_encoding_number = traits.Str(desc="Phase encoding number for reverse DWI")
    reverse_total_readout_time = traits.Str(desc="Total readout time for reverse DWI")

    output_path = Directory(desc="Output directory")

class B0AllAndAcqparamOutputSpec(TraitedSpec):
    acqparam = File(desc="Path to the acqparam.txt file")
    b0_all = File(desc="Path to the b0_all image")

class B0AllAndAcqparam(BaseInterface):
    input_spec = B0AllAndAcqparamInputSpec
    output_spec = B0AllAndAcqparamOutputSpec

    def _run_interface(self, runtime):
        print("Generating b0_all and acqparam.txt...")
        os.mkdir(self.inputs.output_path, exist_ok=True)

        # --- Handle forward DWI ---
        dwi_img = nib.load(self.inputs.dwi_img)
        dwi_data = dwi_img.get_fdata()
        bval = np.loadtxt(self.inputs.dwi_bval)

        b0_indices = np.where(bval < 50)[0]
        print(f"Found {len(b0_indices)} b=0 volumes in DWI")

        b0_vols = dwi_data[..., b0_indices]
        b0_mean = np.mean(b0_vols, axis=3)
        b0_1_path = os.path.join(self.inputs.output_path, "b0_1.nii.gz")
        nib.save(nib.Nifti1Image(b0_mean, dwi_img.affine, dwi_img.header), b0_1_path)

        # --- Handle reverse DWI ---
        rev_img = nib.load(self.inputs.reverse_dwi_img)
        rev_data = rev_img.get_fdata()

        if rev_data.ndim == 3:
            print("Reverse DWI is 3D.")
            b0_2_data = rev_data
        else:
            print("Reverse DWI is 4D.")
            rev_bval = np.loadtxt(self.inputs.reverse_dwi_bval)
            rev_b0_indices = np.where(rev_bval < 50)[0]
            print(f"Found {len(rev_b0_indices)} b=0 volumes in reverse DWI")
            b0_2_data = np.mean(rev_data[..., rev_b0_indices], axis=3)

        b0_2_path = os.path.join(self.inputs.output_path, "b0_2.nii.gz")
        nib.save(nib.Nifti1Image(b0_2_data, rev_img.affine, rev_img.header), b0_2_path)

        # --- Merge b0_1 and b0_2 ---
        subprocess.run(['fslmerge', '-t', os.path.join(self.inputs.output_path, 'b0_all.nii.gz'), b0_1_path, b0_2_path], check=True)

        # --- Write acqparam.txt ---
        with open(os.path.join(self.inputs.output_path, 'acqparam.txt'), 'w') as f:
            f.write(self.inputs.phase_encoding_number + ' ' + str(self.inputs.total_readout_time) + '\n')
            f.write(self.inputs.reverse_phase_encoding_number + ' ' + str(self.inputs.reverse_total_readout_time))

        # --- Cleanup ---
        os.remove(b0_1_path)
        os.remove(b0_2_path)

        self._b0_all = os.path.join(self.inputs.output_path, 'b0_all.nii.gz')
        self._acqparam = os.path.join(self.inputs.output_path, 'acqparam.txt')

        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['acqparam'] = self._acqparam
        outputs['b0_all'] = self._b0_all

        return outputs

######################
# Generate index.txt #
######################

class IndexTxtInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True, desc="Path to the .bval file")
    output_dir = Directory(mandatory=True, desc="Directory to save the index.txt")

class IndexTxtOutputSpec(TraitedSpec):
    index_file = File(desc="Path to the generated index.txt file")

class IndexTxt(BaseInterface):
    input_spec = IndexTxtInputSpec
    output_spec = IndexTxtOutputSpec

    def _run_interface(self, runtime):
        print("Generating index.txt from bval file...")

        bval_path = self.inputs.bval_file
        output_dir = self.inputs.output_dir

        with open(bval_path, 'r') as f:
            bvals = f.read().split()
            b_number = len(bvals)

        print(f"Detected {b_number} b-values")

        index_str = ' '.join(['1'] * b_number)
        index_path = os.path.join(output_dir, "index.txt")
        with open(index_path, 'w') as f:
            f.write(index_str + "\n")

        print(f"index.txt saved to: {index_path}")
        self._index_file = index_path

        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['index_file'] = self._index_file

        return outputs

#########
# TOPUP #
#########

class TopupInputSpec(CommandLineInputSpec):
    b0_all_file = File(exists=True, mandatory=True, desc="Path to the b0_all image", argstr="--imain=%s")
    acqparam_file = File(exists=True, mandatory=True, desc="Path to the acqparam.txt file", argstr="--datain=%s")
    config_file = Str(desc="Path to the config file", argstr="--config=%s")
    output_basename = Str(desc="Path to the output basename", argstr="--out=%s")
    output_b0_basename = Str(desc="Path to the output b0 basename", argstr="--iout=%s")

class TopupOutputSpec(TraitedSpec):
    topup_basename = Str(desc="Path to the topup basename")
    b0_image = File(desc="Path to the b0 image")

class Topup(CommandLine):
    _cmd = 'topup'
    input_spec = TopupInputSpec
    output_spec = TopupOutputSpec
    terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['topup_basename'] = os.path.abspath(self.inputs.output_basename)
        outputs['b0_image'] = os.path.abspath(self.inputs.output_b0_basename + ".nii.gz")

        return outputs
    
###########################
# eddy_cuda10.2 diffusion #
###########################

class EddyCudaInputSpec(CommandLineInputSpec):
    dwi_file = File(exists=True, mandatory=True, desc="Path to the DWI image", argstr="--imain=%s")
    mask_file = File(exists=True, mandatory=True, desc="Path to the brain mask", argstr="--mask=%s")
    bval_file = File(exists=True, mandatory=True, desc="Path to the .bval file", argstr="--bvals=%s")
    bvec_file = File(exists=True, mandatory=True, desc="Path to the .bvec file", argstr="--bvecs=%s")
    index_file = File(exists=True, mandatory=True, desc="Path to the index.txt file", argstr="--index=%s")
    acqparam_file = File(exists=True, mandatory=True, desc="Path to the acqparam.txt file", argstr="--acqp=%s")
    topup_basename = Str(desc="Path to the topup basename", argstr="--topup=%s")
    output_basename = Str(desc="Path to the output basename", argstr="--out=%s")

class EddyCudaOutputSpec(TraitedSpec):
    eddy_output_dir = Directory(desc="Path to the eddy output directory")
    output_basename = Str(desc="Path to the output basename")
    output_filename = Str(desc="Path to the output filename")
    eddy_corrected_data = File(desc="Path to the eddy-corrected DWI image")
    eddy_corrected_bvecs = File(desc="Path to the eddy-corrected bvecs file")
    bvals = File(desc="Path to the bvals file")

# class EddyCuda(CommandLine):
#     _cmd = 'eddy_cuda10.2 diffusion'
#     input_spec = EddyCudaInputSpec
#     output_spec = EddyCudaOutputSpec
#     terminal_output = 'allatonce'

#     def _list_outputs(self):
#         outputs = self.output_spec().get()
#         outputs['eddy_corrected_data'] = os.path.abspath(self.inputs.output_basename + ".nii.gz")
#         outputs['eddy_corrected_bvecs'] = os.path.abspath(self.inputs.output_basename + ".eddy_rotated_bvecs")
#         outputs['bvals'] = os.path.abspath(self.inputs.bval_file)
#         outputs['dwi_b0_brain_mask'] = os.path.abspath(self.inputs.mask_file)

#         return outputs

class EddyCuda(CommandLine):
    _cmd = 'eddy_cuda10.2 diffusion'
    input_spec = EddyCudaInputSpec
    output_spec = EddyCudaOutputSpec
    terminal_output = 'allatonce'

    def _run_interface(self, runtime):
        output_path = os.path.abspath(self.inputs.output_basename + ".nii.gz")
        if os.path.exists(output_path):
            runtime.returncode = 0
            runtime.stdout = f"{output_path} exists, skipping eddy_cuda."
            runtime.stderr = ""
            return runtime

        return super()._run_interface(runtime)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['eddy_output_dir'] = os.path.abspath(os.path.dirname(self.inputs.output_basename))
        outputs['output_basename'] = self.inputs.output_basename
        outputs['output_filename'] = os.path.basename(self.inputs.output_basename)
        outputs['eddy_corrected_data'] = os.path.abspath(self.inputs.output_basename + ".nii.gz")
        outputs['eddy_corrected_bvecs'] = os.path.abspath(self.inputs.output_basename + ".eddy_rotated_bvecs")
        outputs['bvals'] = os.path.abspath(self.inputs.bval_file)
        return outputs
    
######################
# Order eddy outputs #
######################
class OrderEddyOutputsInputSpec(CommandLineInputSpec):
    eddy_output_dir = Directory(exists=True, mandatory=True, desc="Path to the eddy output directory", argstr="%s", position=0)
    eddy_output_filename = Str(desc="Base name for the eddy output files", argstr="%s", position=1)
    new_output_dir = Str(desc="Directory to save the ordered outputs", argstr="%s", position=2)
    new_output_filename = Str(desc="Base name for the new output files", argstr="%s", position=3)
    bval = File(exists=True, desc="Path to the bval file", argstr="%s", position=4)
    output_resolution = Float(desc="Output resolution in mm (isotropic)", argstr="%f", position=5)

class OrderEddyOutputsOutputSpec(TraitedSpec):
    ordered_dwi = File(desc="Path to the ordered DWI image")
    ordered_bvec = File(desc="Path to the ordered bvec file")
    ordered_bval = File(desc="Path to the ordered bval file")

class OrderEddyOutputs(CommandLine):
    input_spec = OrderEddyOutputsInputSpec
    output_spec = OrderEddyOutputsOutputSpec
    _cmd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bash", "fdt", "fdt_order_eddyout.sh"))

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['ordered_dwi'] = os.path.join(self.inputs.new_output_dir, self.inputs.new_output_filename + ".nii.gz")
        outputs['ordered_bvec'] = os.path.join(self.inputs.new_output_dir, self.inputs.new_output_filename + ".bvec")
        outputs['ordered_bval'] = os.path.join(self.inputs.new_output_dir, self.inputs.new_output_filename + ".bval")
        return outputs

#########################
# B0 ref and brain mask #
#########################
class B0RefAndBrainMaskInputSpec(CommandLineInputSpec):
    # Usage: $0 <input_dwi> <input_bval> <output_dir> <output_b0_filename> <output_b0_mask_filename>
    input_dwi = File(exists=True, mandatory=True, desc="Path to the input DWI image", argstr="%s", position=0)
    input_bval = File(exists=True, mandatory=True, desc="Path to the input bval file", argstr="%s", position=1)
    output_dir = Str(desc="Directory to save the outputs", argstr="%s", position=2)
    output_b0_filename = Str(desc="Base name for the output b0 image", argstr="%s", position=3)
    output_b0_mask_filename = Str(desc="Base name for the output b0 brain mask", argstr="%s", position=4)
class B0RefAndBrainMaskOutputSpec(TraitedSpec):
    b0_image = File(desc="Path to the b0 image")
    b0_brain_mask = File(desc="Path to the b0 brain mask")
class B0RefAndBrainMask(CommandLine):
    _cmd = 'bash ' + os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bash", "fdt", "extract_b0.sh"))
    input_spec = B0RefAndBrainMaskInputSpec
    output_spec = B0RefAndBrainMaskOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['b0_image'] = os.path.join(self.inputs.output_dir, self.inputs.output_b0_filename)
        outputs['b0_brain_mask'] = os.path.join(self.inputs.output_dir, self.inputs.output_b0_mask_filename)
        return outputs

###########################
# DTIFit with BIDS rename #
###########################
class DTIFitBIDSInputSpec(CommandLineInputSpec):
    dwi_file = File(exists=True, mandatory=True, desc="Path to the DWI image", argstr="-k %s")
    bval_file = File(exists=True, mandatory=True, desc="Path to the .bval file", argstr="-b %s")
    bvec_file = File(exists=True, mandatory=True, desc="Path to the .bvec file", argstr="-r %s")
    mask_file = File(exists=True, mandatory=True, desc="Path to the brain mask", argstr="-m %s")

    # This is the dtifit "-o" basename. We still use it as a temporary basename.
    output_basename = Str(mandatory=True, desc="Path to the output basename", argstr="-o %s")

    # Whether to rename outputs into BIDS style after dtifit
    bids_rename = traits.Bool(True, usedefault=True, desc="Rename outputs to BIDS style after dtifit")

    # Safety option: overwrite destination files if exist
    overwrite = traits.Bool(False, usedefault=True, desc="Overwrite destination files if they already exist")


class DTIFitBIDSOutputSpec(TraitedSpec):
    output_dir = Directory(desc="Path to the output directory")
    output_basename = Str(desc="Path to the output basename")

    dti_fa = File(desc="Path to the FA image")
    dti_md = File(desc="Path to the MD image")
    dti_mo = File(desc="Path to the MO image")
    dti_tensor = File(desc="Path to the tensor image")

    dti_l1 = File(desc="Path to the L1 image")
    dti_l2 = File(desc="Path to the L2 image")
    dti_l3 = File(desc="Path to the L3 image")

    dti_v1 = File(desc="Path to the V1 image")
    dti_v2 = File(desc="Path to the V2 image")
    dti_v3 = File(desc="Path to the V3 image")

    dti_s0 = File(desc="Path to the S0 image")

class DTIFitBIDS(CommandLine):
    """
    Run FSL dtifit and (optionally) rename outputs to BIDS style.
    """
    _cmd = "dtifit"
    input_spec = DTIFitBIDSInputSpec
    output_spec = DTIFitBIDSOutputSpec
    terminal_output = "allatonce"

    def _ensure_outdir(self):
        out_dir = os.path.abspath(os.path.dirname(self.inputs.output_basename))
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _dtifit_expected_files(self):
        """
        dtifit uses uppercase suffixes for most outputs:
          <base>_FA.nii.gz, <base>_MD.nii.gz, <base>_MO.nii.gz,
          <base>_L1.nii.gz, <base>_L2.nii.gz, <base>_L3.nii.gz,
          <base>_V1.nii.gz, <base>_V2.nii.gz, <base>_V3.nii.gz,
          <base>_S0.nii.gz, <base>_tensor.nii.gz
        """
        base = os.path.abspath(self.inputs.output_basename)
        mapping = {
            "fa": f"{base}_FA.nii.gz",
            "md": f"{base}_MD.nii.gz",
            "mo": f"{base}_MO.nii.gz",
            "l1": f"{base}_L1.nii.gz",
            "l2": f"{base}_L2.nii.gz",
            "l3": f"{base}_L3.nii.gz",
            "v1": f"{base}_V1.nii.gz",
            "v2": f"{base}_V2.nii.gz",
            "v3": f"{base}_V3.nii.gz",
            "s0": f"{base}_S0.nii.gz",
            "tensor": f"{base}_tensor.nii.gz",
        }
        return mapping

    def _bids_dest_files(self):
        from cvdproc.bids_data.rename_bids_file import rename_bids_file

        out_dir = os.path.abspath(os.path.dirname(self.inputs.output_basename))
        dest = {}
        for param in ["fa", "md", "mo", "tensor", "l1", "l2", "l3", "v1", "v2", "v3", "s0"]:
            fname = rename_bids_file(
                self.inputs.dwi_file,
                {"desc": None, "model": "tensor", "param": param},
                "dwimap",
                ".nii.gz",
            )
            dest[param] = os.path.join(out_dir, fname)
        return dest

    def _run_interface(self, runtime):
        self._ensure_outdir()

        # 1) Run dtifit
        runtime = super()._run_interface(runtime)

        # 2) Rename/move outputs into BIDS style
        if bool(self.inputs.bids_rename):
            src = self._dtifit_expected_files()
            dst = self._bids_dest_files()

            for param, src_path in src.items():
                if not os.path.exists(src_path):
                    raise FileNotFoundError(f"dtifit output not found: {src_path}")

                dst_path = dst[param]
                if os.path.exists(dst_path):
                    if bool(self.inputs.overwrite):
                        os.remove(dst_path)
                    else:
                        raise FileExistsError(f"Destination exists: {dst_path}")

                shutil.move(src_path, dst_path)

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["output_dir"] = os.path.abspath(os.path.dirname(self.inputs.output_basename))
        outputs["output_basename"] = self.inputs.output_basename

        if bool(self.inputs.bids_rename):
            dst = self._bids_dest_files()
            outputs["dti_fa"] = dst["fa"]
            outputs["dti_md"] = dst["md"]
            outputs["dti_mo"] = dst["mo"]
            outputs["dti_tensor"] = dst["tensor"]
            outputs["dti_l1"] = dst["l1"]
            outputs["dti_l2"] = dst["l2"]
            outputs["dti_l3"] = dst["l3"]
            outputs["dti_v1"] = dst["v1"]
            outputs["dti_v2"] = dst["v2"]
            outputs["dti_v3"] = dst["v3"]
            outputs["dti_s0"] = dst["s0"]
        else:
            src = self._dtifit_expected_files()
            outputs["dti_fa"] = src["fa"]
            outputs["dti_md"] = src["md"]
            outputs["dti_mo"] = src["mo"]
            outputs["dti_tensor"] = src["tensor"]
            outputs["dti_l1"] = src["l1"]
            outputs["dti_l2"] = src["l2"]
            outputs["dti_l3"] = src["l3"]
            outputs["dti_v1"] = src["v1"]
            outputs["dti_v2"] = src["v2"]
            outputs["dti_v3"] = src["v3"]
            outputs["dti_s0"] = src["s0"]

        return outputs
    
############
# bedpostx #
############

class PrepareBedpostxInputSpec(BaseInterfaceInputSpec):
    dwi_img = File(exists=True, desc='DWI image')
    bvec = File(exists=True, desc='Bvec file')
    bval = File(exists=True, desc='Bval file')
    mask = File(exists=True, desc='Mask file')
    output_dir = Directory(exists=True, desc='Output directory')
    use_symlinks = Bool(True, desc='Use symlinks instead of copying files')

class PrepareBedpostxOutputSpec(TraitedSpec):
    bedpostx_input_dir = Directory(desc='Bedpostx input directory')

class PrepareBedpostx(BaseInterface):
    input_spec = PrepareBedpostxInputSpec
    output_spec = PrepareBedpostxOutputSpec

    def _run_interface(self, runtime):
        dwi_img = self.inputs.dwi_img
        bvec = self.inputs.bvec
        bval = self.inputs.bval
        mask = self.inputs.mask
        output_dir = self.inputs.output_dir

        # Create the bedpostx input directory
        bedpostx_input_dir = output_dir
        os.makedirs(bedpostx_input_dir, exist_ok=True)

        if self.inputs.use_symlinks:
            # Create symlinks to the files in the bedpostx input directory
            os.symlink(os.path.abspath(dwi_img), os.path.join(bedpostx_input_dir, 'data.nii.gz'))
            os.symlink(os.path.abspath(bvec), os.path.join(bedpostx_input_dir, 'bvecs'))
            os.symlink(os.path.abspath(bval), os.path.join(bedpostx_input_dir, 'bvals'))
            os.symlink(os.path.abspath(mask), os.path.join(bedpostx_input_dir, 'nodif_brain_mask.nii.gz'))
        else:
            # Copy the files to the bedpostx input directory
            shutil.copy(dwi_img, os.path.join(bedpostx_input_dir, 'data.nii.gz'))
            shutil.copy(bvec, os.path.join(bedpostx_input_dir, 'bvecs'))
            shutil.copy(bval, os.path.join(bedpostx_input_dir, 'bvals'))
            shutil.copy(mask, os.path.join(bedpostx_input_dir, 'nodif_brain_mask.nii.gz'))

        self._bedpostx_input_dir = bedpostx_input_dir

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['bedpostx_input_dir'] = self._bedpostx_input_dir

        return outputs

class BedpostxInputSpec(CommandLineInputSpec):
    input_dir = Str(argstr='%s', position=0, desc='Input directory', mandatory=True)

class BedpostxOutputSpec(TraitedSpec):
    output_dir = Str(desc='Output directory')

class Bedpostx(CommandLine):
    #_cmd = 'bedpostx_gpu'
    _cmd = 'bash ' + get_package_path('pipelines', 'bash', 'fdt', 'bedpostx_gpu_custom.sh')
    input_spec = BedpostxInputSpec
    output_spec = BedpostxOutputSpec
    #terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_dir'] = self.inputs.input_dir + '.bedpostX'

        return outputs

class BedpostxGPUCustomInputSpec(CommandLineInputSpec):
    # Required inputs
    dwi_img = File(
        exists=True,
        mandatory=True,
        desc="Path to DWI image (4D)",
        argstr="--dwi %s",
    )
    bvec = File(
        exists=True,
        mandatory=True,
        desc="Path to bvecs",
        argstr="--bvec %s",
    )
    bval = File(
        exists=True,
        mandatory=True,
        desc="Path to bvals",
        argstr="--bval %s",
    )
    mask = File(
        exists=True,
        mandatory=True,
        desc="Path to nodif_brain_mask image",
        argstr="--mask %s",
    )
    out_dir = Directory(
        mandatory=True,
        desc="Output directory (final results; no .bedpostX suffix)",
        argstr="--out-dir %s",
    )

    # Optional bedpostx-like options supported by your script
    njobs = Int(
        4,
        usedefault=True,
        desc="Number of jobs/parts",
        argstr="-NJOBS %d",
    )
    nfibres = Int(
        3,
        usedefault=True,
        desc="Number of fibres per voxel",
        argstr="-n %d",
    )
    fudge = Float(
        1.0,
        usedefault=True,
        desc="ARD weight/fudge",
        argstr="-w %f",
    )
    burnin = Int(
        1000,
        usedefault=True,
        desc="Burnin",
        argstr="-b %d",
    )
    njumps = Int(
        1250,
        usedefault=True,
        desc="Number of jumps",
        argstr="-j %d",
    )
    sampleevery = Int(
        25,
        usedefault=True,
        desc="Sample every",
        argstr="-s %d",
    )
    model = Int(
        2,
        usedefault=True,
        desc="Model: 1 sticks, 2 sticks+range, 3 zeppelins",
        argstr="-model %d",
    )

    # Gradient nonlinearity support (only if you implemented --grad-dev in the script)
    grad_dev = File(
        exists=True,
        mandatory=False,
        desc="Path to grad_dev image (optional)",
        argstr="--grad-dev %s",
    )

    # Additional xfibres options to pass through, e.g. ["--noard", "--cnonlinear"]
    # Nipype's CommandLine supports "args" as a raw string, so we provide an explicit field.
    extra_args = Str(
        "",
        usedefault=True,
        desc="Extra arguments passed to xfibres_gpu/bedpostx (raw string)",
        argstr="%s",
    )


class BedpostxGPUCustomOutputSpec(TraitedSpec):
    out_dir = Directory(desc="Output directory (final bedpostx results)")
    eye_mat = File(desc="Identity transform matrix")
    diff_parts_dir = Directory(desc="diff_parts directory")
    logs_dir = Directory(desc="logs directory")
    xfms_dir = Directory(desc="xfms directory")
    source_for_probtrackx = Str(desc="Source string for probtrackx input (-s argument)")
    mask_for_probtrackx = Str(desc="Mask string for probtrackx input (-m argument)")

class BedpostxGPUCustom(CommandLine):
    """
    Run modified bedpostx_gpu_custom.sh that accepts explicit input files and a user-defined output directory.
    """
    _cmd = "bash " + get_package_path("pipelines", "bash", "fdt", "bedpostx_gpu_custom.sh")
    input_spec = BedpostxGPUCustomInputSpec
    output_spec = BedpostxGPUCustomOutputSpec
    terminal_output = "allatonce"

    def _run_interface(self, runtime):
        # Ensure output directory exists before execution (script also does, but safe here)
        out_dir = os.path.abspath(self.inputs.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        return super()._run_interface(runtime)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        out_dir = os.path.abspath(self.inputs.out_dir)

        outputs["out_dir"] = out_dir
        outputs["eye_mat"] = os.path.join(out_dir, "xfms", "eye.mat")
        outputs["diff_parts_dir"] = os.path.join(out_dir, "diff_parts")
        outputs["logs_dir"] = os.path.join(out_dir, "logs")
        outputs["xfms_dir"] = os.path.join(out_dir, "xfms")
        outputs["source_for_probtrackx"] = os.path.join(out_dir, "merged")
        outputs["mask_for_probtrackx"] = os.path.join(out_dir, "nodif_brain_mask.nii.gz")

        return outputs

##############
# Probtrackx #
##############
class ProbtrackxInputSpec(CommandLineInputSpec):
    source = Str(argstr='-s %s', desc='Source bedpostx output', mandatory=True)
    dwi_mask = File(argstr='-m %s', desc='DWI brain mask', mandatory=True)
    seed_mask = File(argstr='-x %s', desc='Seed mask', mandatory=True)
    seed_to_dwi_xfm = File(argstr='--xfm=%s', desc='Transform from seed to DWI space', mandatory=True)
    seed_ref = File(argstr='--seedref=%s', desc='Seed reference image', mandatory=True)
    waypoints = Str(argstr='--waypoints=%s', desc='Waypoints file', mandatory=False)
    path_length_correction = Bool(False, argstr='--pd', desc='Enable path length correction', mandatory=False)
    output_dir = Str(argstr='--dir=%s', desc='Output directory', mandatory=True)
    nsamples = Int(5000, argstr='--nsamples=%d', desc='Number of samples', mandatory=False)
    args = Str(argstr='%s', desc='Additional arguments', mandatory=False)

class ProbtrackxOutputSpec(TraitedSpec):
    output_dir = Str(desc='Output directory')
    fdt_paths = File(desc='FDT paths file')

class Probtrackx(CommandLine):
    # use gpu
    _cmd = 'probtrackx2_gpu'
    input_spec = ProbtrackxInputSpec
    output_spec = ProbtrackxOutputSpec
    terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_dir'] = os.path.abspath(self.inputs.output_dir)
        outputs['fdt_paths'] = os.path.abspath(os.path.join(self.inputs.output_dir, 'fdt_paths.nii.gz'))
        return outputs

###############    
# Merge GIFTI #
###############
class MergeGiftiInputSpec(BaseInterfaceInputSpec):
    gii_mesh = File(exists=True, desc='GIFTI mesh file (for vertices and faces)')
    gii_data = File(exists=True, desc='GIFTI data file (for data values)')
    output_gii = File(desc='Output GIFTI file')

class MergeGiftiOutputSpec(TraitedSpec):
    output_gii = File(desc='Output GIFTI file')

class MergeGifti(BaseInterface):
    input_spec = MergeGiftiInputSpec
    output_spec = MergeGiftiOutputSpec

    def _run_interface(self, runtime):
        gii_mesh_path = self.inputs.gii_mesh
        gii_data_path = self.inputs.gii_data
        output_gii = self.inputs.output_gii

        # rewrite _merge_wmgm_boundary_gifti
        vertices = nib.load(gii_mesh_path).darrays[0].data
        faces = nib.load(gii_mesh_path).darrays[1].data
        values = nib.load(gii_data_path).darrays[0].data

        num_vertices = vertices.shape[0]
        assert values.shape[0] == num_vertices, "Error: Measure and mesh vertex counts do not match!"

        new_gii = nib.gifti.GiftiImage()
        new_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(
            vertices, intent=nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']
        ))
        new_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(
            faces, intent=nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']
        ))
        new_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(
            values, intent=nib.nifti1.intent_codes['NIFTI_INTENT_SHAPE']
        ))

        nib.save(new_gii, output_gii)
        print(f"Merged GIFTI file saved as {output_gii}")

        self._output_gii = output_gii

        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_gii'] = self._output_gii

        return outputs

###########################
# Apply GIFTI mask to MGH #
###########################
class ApplyGiiMaskToMghInputSpec(BaseInterfaceInputSpec):
    measure_mgh = File(exists=True, mandatory=True, desc="Measure MGH file")
    mask_gii = File(exists=True, mandatory=True, desc="Mask GIFTI file")
    output_mgh = File(mandatory=True, desc="Output masked MGH file")

class ApplyGiiMaskToMghOutputSpec(TraitedSpec):
    output_mgh = File(exists=True, desc="Output masked MGH file")

class ApplyGiiMaskToMgh(BaseInterface):
    input_spec = ApplyGiiMaskToMghInputSpec
    output_spec = ApplyGiiMaskToMghOutputSpec

    def _run_interface(self, runtime):
        measure_mgh_path = self.inputs.measure_mgh
        mask_gii_path = self.inputs.mask_gii
        output_mgh_path = self.inputs.output_mgh

        measure_img = nib.load(measure_mgh_path)
        measure_data = np.asanyarray(measure_img.get_fdata()).squeeze()

        if measure_data.ndim != 1:
            raise ValueError(f"Expected 1D measure data after squeeze, got shape {measure_data.shape}")

        mask_gii = nib.load(mask_gii_path)
        darrays = getattr(mask_gii, "darrays", None)
        if darrays is None or len(darrays) == 0:
            raise ValueError(f"No darrays found in GIFTI file: {mask_gii_path}")

        target_len = int(measure_data.shape[0])

        candidates = []
        for i, da in enumerate(darrays):
            data = np.asanyarray(da.data).squeeze()
            if data.ndim == 1:
                candidates.append((i, data))

        if len(candidates) == 0:
            raise ValueError(f"No 1D darray found in GIFTI file: {mask_gii_path}")

        mask_data = None

        # Prefer exact length match to the MGH vector
        for i, data in candidates:
            if int(data.shape[0]) == target_len:
                mask_data = data
                break

        # Fallback: if only one 1D darray exists, use it but still validate
        if mask_data is None and len(candidates) == 1:
            mask_data = candidates[0][1]

        if mask_data is None:
            shapes = [(i, c.shape) for i, c in candidates]
            raise ValueError(
                f"Could not find a 1D GIFTI darray matching measure length {target_len}. "
                f"Candidates: {shapes}. File: {mask_gii_path}"
            )

        if int(mask_data.shape[0]) != target_len:
            raise ValueError(
                f"Mask length mismatch: measure length {target_len} vs mask length {int(mask_data.shape[0])} "
                f"(mask file: {mask_gii_path})"
            )

        unique_vals = np.unique(mask_data)
        if not np.all(np.isin(unique_vals, [0, 1])):
            mask_data = (mask_data > 0).astype(measure_data.dtype)

        masked_data = measure_data * mask_data.astype(measure_data.dtype)

        out_dir = os.path.dirname(os.path.abspath(output_mgh_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        masked_img = nib.MGHImage(masked_data, measure_img.affine, measure_img.header)
        nib.save(masked_img, output_mgh_path)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_mgh"] = os.path.abspath(self.inputs.output_mgh)
        return outputs

#################################
# Apply GIFTI mask to MGH (TXT) #
#################################

class ApplyIndexTxtToMghInputSpec(BaseInterfaceInputSpec):
    measure_mgh = File(exists=True, mandatory=True, desc="Input surface measure MGH (1D vector)")
    index_txt = File(exists=True, mandatory=True, desc="Text file containing vertex indices to zero")
    output_mgh = File(mandatory=True, desc="Output masked MGH file")
    index_base = traits.Enum(0, 1, usedefault=True, desc="0 if indices are 0-based, 1 if indices are 1-based")

class ApplyIndexTxtToMghOutputSpec(TraitedSpec):
    output_mgh = File(exists=True, desc="Output masked MGH file")

class ApplyIndexTxtToMgh(BaseInterface):
    input_spec = ApplyIndexTxtToMghInputSpec
    output_spec = ApplyIndexTxtToMghOutputSpec

    def _read_indices(self, txt_path: str) -> np.ndarray:
        indices = []
        with open(txt_path, "r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    continue
                try:
                    indices.append(int(s))
                except ValueError:
                    # Ignore non-integer lines safely
                    continue
        if len(indices) == 0:
            raise ValueError(f"No valid integer indices found in: {txt_path}")
        idx = np.asarray(indices, dtype=np.int64)
        if self.inputs.index_base == 1:
            idx = idx - 1
        return idx

    def _run_interface(self, runtime):
        measure_mgh_path = self.inputs.measure_mgh
        index_txt_path = self.inputs.index_txt
        output_mgh_path = self.inputs.output_mgh

        measure_img = nib.load(measure_mgh_path)
        measure_data = np.asanyarray(measure_img.get_fdata()).squeeze()

        if measure_data.ndim != 1:
            raise ValueError(f"Expected 1D measure data after squeeze, got shape {measure_data.shape}")

        n = int(measure_data.shape[0])
        idx = self._read_indices(index_txt_path)

        if np.any(idx < 0) or np.any(idx >= n):
            bad = idx[(idx < 0) | (idx >= n)]
            preview = bad[:10].tolist()
            raise ValueError(
                f"Index out of range for measure length {n}. "
                f"Bad indices (first 10): {preview}. "
                f"Check index_base={int(self.inputs.index_base)} and the txt file."
            )

        masked = measure_data.copy()
        masked[idx] = 0

        out_dir = os.path.dirname(os.path.abspath(output_mgh_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        out_img = nib.MGHImage(masked, measure_img.affine, measure_img.header)
        nib.save(out_img, output_mgh_path)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_mgh"] = os.path.abspath(self.inputs.output_mgh)
        return outputs

###########################
# Define Connection-level #
###########################
class DefineConnectionLevelInputSpec(BaseInterfaceInputSpec):
    lh_mgh = File(exists=True, desc='Left hemisphere MGH file')
    rh_mgh = File(exists=True, desc='Right hemisphere MGH file')
    output_files = Dict(desc='Dictionary of output file paths for each connection level, should contain: \'lh_unconn\', \'rh_unconn\', \'lh_lowconn\', \'rh_lowconn\', \'lh_medconn\', \'rh_medconn\', \'lh_highconn\', \'rh_highconn\'')
    low_threshold = Float(desc='Low threshold value')
    divisor = Float(1, desc='Divisor for normalization')

class DefineConnectionLevelOutputSpec(TraitedSpec):
    output_files = List(File, desc='List of output files for each connection level')
    lh_unconn_mask = File(desc='Left hemisphere unconnected mask')
    rh_unconn_mask = File(desc='Right hemisphere unconnected mask')
    lh_lowconn_mask = File(desc='Left hemisphere low connectivity mask')
    rh_lowconn_mask = File(desc='Right hemisphere low connectivity mask')
    lh_medconn_mask = File(desc='Left hemisphere medium connectivity mask')
    rh_medconn_mask = File(desc='Right hemisphere medium connectivity mask')
    lh_highconn_mask = File(desc='Left hemisphere high connectivity mask')
    rh_highconn_mask = File(desc='Right hemisphere high connectivity mask')

class DefineConnectionLevel(BaseInterface):
    input_spec = DefineConnectionLevelInputSpec
    output_spec = DefineConnectionLevelOutputSpec

    def _run_interface(self, runtime):
        lh_mgh_path = self.inputs.lh_mgh
        rh_mgh_path = self.inputs.rh_mgh
        output_files = self.inputs.output_files
        low_threshold = self.inputs.low_threshold
        divisor = self.inputs.divisor

        # Load data
        lh_data = nib.load(lh_mgh_path).get_fdata().squeeze()
        rh_data = nib.load(rh_mgh_path).get_fdata().squeeze()

        # Normalize data
        lh_data_norm = lh_data / divisor
        rh_data_norm = rh_data / divisor

        combined_data = np.concatenate((lh_data_norm, rh_data_norm))
        filtered_data = combined_data[combined_data > low_threshold]
        sorted_values = np.sort(filtered_data)

        # Percentage thresholds
        medium_threshold = np.percentile(sorted_values, 50)
        high_threshold = np.percentile(sorted_values, 75)

        lh_unconn_mask = (lh_data_norm <= low_threshold).astype(np.float32)
        rh_unconn_mask = (rh_data_norm <= low_threshold).astype(np.float32)
        lh_lowconn_mask = (lh_data_norm > low_threshold).astype(np.float32)
        rh_lowconn_mask = (rh_data_norm > low_threshold).astype(np.float32)
        lh_medconn_mask = (lh_data_norm > medium_threshold).astype(np.float32)
        rh_medconn_mask = (rh_data_norm > medium_threshold).astype(np.float32)
        lh_highconn_mask = (lh_data_norm > high_threshold).astype(np.float32)
        rh_highconn_mask = (rh_data_norm > high_threshold).astype(np.float32)

        def save_mgh(data, ref_img, filename):
            img = nib.MGHImage(data.astype(np.float32), ref_img.affine, ref_img.header)
            nib.save(img, filename)
        
        save_mgh(lh_unconn_mask, nib.load(lh_mgh_path), output_files['lh_unconn'])
        save_mgh(rh_unconn_mask, nib.load(rh_mgh_path), output_files['rh_unconn'])
        save_mgh(lh_lowconn_mask, nib.load(lh_mgh_path), output_files['lh_lowconn'])
        save_mgh(rh_lowconn_mask, nib.load(rh_mgh_path), output_files['rh_lowconn'])
        save_mgh(lh_medconn_mask, nib.load(lh_mgh_path), output_files['lh_medconn'])
        save_mgh(rh_medconn_mask, nib.load(rh_mgh_path), output_files['rh_medconn'])
        save_mgh(lh_highconn_mask, nib.load(lh_mgh_path), output_files['lh_highconn'])
        save_mgh(rh_highconn_mask, nib.load(rh_mgh_path), output_files['rh_highconn'])
        self._output_files = list(output_files.values())
        self._lh_unconn_mask = output_files['lh_unconn']
        self._rh_unconn_mask = output_files['rh_unconn']
        self._lh_lowconn_mask = output_files['lh_lowconn']
        self._rh_lowconn_mask = output_files['rh_lowconn']
        self._lh_medconn_mask = output_files['lh_medconn']
        self._rh_medconn_mask = output_files['rh_medconn']
        self._lh_highconn_mask = output_files['lh_highconn']
        self._rh_highconn_mask = output_files['rh_highconn']

        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_files'] = self._output_files
        outputs['lh_unconn_mask'] = self._lh_unconn_mask
        outputs['rh_unconn_mask'] = self._rh_unconn_mask
        outputs['lh_lowconn_mask'] = self._lh_lowconn_mask
        outputs['rh_lowconn_mask'] = self._rh_lowconn_mask
        outputs['lh_medconn_mask'] = self._lh_medconn_mask
        outputs['rh_medconn_mask'] = self._rh_medconn_mask
        outputs['lh_highconn_mask'] = self._lh_highconn_mask
        outputs['rh_highconn_mask'] = self._rh_highconn_mask

        return outputs

##############################
# Extract Surface Parameters #
##############################
# nipype interface of mri_segstats
class MRIsegstatsInputSpec(CommandLineInputSpec):
    segvol = Str(argstr='--seg %s', position=0, desc='Segmentation volume', mandatory=True)
    invol = Str(argstr='--in %s', position=1, desc='Input volume', mandatory=True)
    sum = Str(argstr='--sum %s', position=2, desc='Output summary file', mandatory=True)

class MRIsegstatsOutputSpec(TraitedSpec):
    sum = Str(desc='Output summary file')

class MRIsegstats(CommandLine):
    _cmd = 'mri_segstats'
    input_spec = MRIsegstatsInputSpec
    output_spec = MRIsegstatsOutputSpec
    terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['sum'] = self.inputs.sum

        return outputs

####################################################################
# Extracrt Surface Parameters (specifically for Tractography Node) #
####################################################################
class ExtractSurfaceParametersInputSpec(BaseInterfaceInputSpec):
    fs_subjects_dir = Directory(exists=True, desc='Freesurfer subjects directory')
    sessions = traits.List(desc='List of sessions')
    lh_unconn_mask = File(exists=True, desc='Unconnected mask (left hemisphere)')
    rh_unconn_mask = File(exists=True, desc='Unconnected mask (right hemisphere)')
    lh_low_conn_mask = File(exists=True, desc='Low connectivity mask (left hemisphere)')
    rh_low_conn_mask = File(exists=True, desc='Low connectivity mask (right hemisphere)')
    lh_medium_conn_mask = File(exists=True, desc='Medium connectivity mask (left hemisphere)')
    rh_medium_conn_mask = File(exists=True, desc='Medium connectivity mask (right hemisphere)')
    lh_high_conn_mask = File(exists=True, desc='High connectivity mask (left hemisphere)')
    rh_high_conn_mask = File(exists=True, desc='High connectivity mask (right hemisphere)')
    output_dir = Str(desc='Output directory')
    csv_file_name = Str(desc='Output CSV file name', default_value='surface_parameters.csv')

class ExtractSurfaceParametersOutputSpec(TraitedSpec):
    csv_file = File(desc='Output CSV file')

class ExtractSurfaceParameters(BaseInterface):
    input_spec = ExtractSurfaceParametersInputSpec
    output_spec = ExtractSurfaceParametersOutputSpec

    def _run_interface(self, runtime):
        fs_subjects_dir = self.inputs.fs_subjects_dir
        sessions = self.inputs.sessions
        output_dir = self.inputs.output_dir

        paired_roi_masks = {
            "unconn": [self.inputs.lh_unconn_mask, self.inputs.rh_unconn_mask],
            "low_conn": [self.inputs.lh_low_conn_mask, self.inputs.rh_low_conn_mask],
            "medium_conn": [self.inputs.lh_medium_conn_mask, self.inputs.rh_medium_conn_mask],
            "high_conn": [self.inputs.lh_high_conn_mask, self.inputs.rh_high_conn_mask],
        }

        fwhm_values = [0, 5, 10, 15, 20, 25]

        measure_files = [
            "area", "area.pial", "curv", "jacobian_white", "sulc",
            "thickness", "volume", "w-g.pct.mgh", "white.H", "white.K"
        ]

        results = []

        # walk through sessions
        for session in sessions:
            session_surf_dir = os.path.join(fs_subjects_dir, f"ses-{session}", 'surf')
            session_data = {"session": f'ses-{session}'}

            if not os.path.exists(session_surf_dir):
                continue
            
            for measure in measure_files:
                for fwhm in fwhm_values:
                    lh_measure_path = os.path.join(session_surf_dir, f"lh.{measure}.fwhm{fwhm}.fsaverage.mgh")
                    rh_measure_path = os.path.join(session_surf_dir, f"rh.{measure}.fwhm{fwhm}.fsaverage.mgh")

                    for roi_name, roi_masks in paired_roi_masks.items():
                        lh_roi_mask = roi_masks[0]
                        rh_roi_mask = roi_masks[1]

                        lh_measure_img = nib.load(lh_measure_path)
                        rh_measure_img = nib.load(rh_measure_path)
                        lh_roi_img = nib.load(lh_roi_mask)
                        rh_roi_img = nib.load(rh_roi_mask)

                        lh_measure_data = lh_measure_img.get_fdata().squeeze()  # (N,)
                        rh_measure_data = rh_measure_img.get_fdata().squeeze()  # (N,)
                        lh_roi_data = lh_roi_img.get_fdata().squeeze()          # (N,)
                        rh_roi_data = rh_roi_img.get_fdata().squeeze()          # (N,)

                        if not np.all(np.equal(np.mod(lh_roi_data, 1), 0)) or not np.all(np.equal(np.mod(rh_roi_data, 1), 0)):
                            raise ValueError("ROI MGH files contain non-integer values. Please check the input files.")
                        
                        combined_measure_data = np.concatenate([lh_measure_data, rh_measure_data])
                        combined_roi_data = np.concatenate([lh_roi_data, rh_roi_data])

                        unique_rois = np.unique(combined_roi_data)
                        roi_avg_values = {}

                        for roi in unique_rois:
                            roi_mask = combined_roi_data == roi
                            roi_avg_values[int(roi)] = np.mean(combined_measure_data[roi_mask])
                        
                        mean_roi_1 = roi_avg_values.get(1, np.nan)
                        colname = f"{roi_name}_{measure}.fwhm{fwhm}"
                        session_data[colname] = mean_roi_1

            results.append(session_data)

        df = pd.DataFrame(results)
        self._csv_file = os.path.join(output_dir, self.inputs.csv_file_name)
        df.to_csv(self._csv_file, index=False)

        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['csv_file'] = self._csv_file

        return outputs