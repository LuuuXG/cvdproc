import os
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec, CommandLineInputSpec, CommandLine,
    File, Str
)
import nibabel as nib
import numpy as np
from scipy.ndimage import (
    generate_binary_structure,
    binary_dilation,
    label as cc_label,
)

from cvdproc.config.paths import get_package_path

# Remove white matter hyperintensities (WMH) from FreeSurfer aseg segmentation
# and fix misclassified choroid plexus labels (31 and 63) based on their contact
# with ventricles vs. white matter.
class RemoveWMHInputSpec(BaseInterfaceInputSpec):
    in_aseg = File(exists=True, mandatory=True)
    out_aseg = Str(mandatory=True)

class RemoveWMHOutputSpec(TraitedSpec):
    out_aseg = File()

class RemoveWMH(BaseInterface):
    input_spec = RemoveWMHInputSpec
    output_spec = RemoveWMHOutputSpec

    def _run_interface(self, runtime):

        in_aseg_path = self.inputs.in_aseg
        out_aseg_path = self.inputs.out_aseg

        aseg_img = nib.load(in_aseg_path)
        aseg_data = aseg_img.get_fdata().astype(np.int16)

        # ---------------------------------
        # Step 1: Remove WMH (label = 77)
        # ---------------------------------
        wmh_mask = aseg_data == 77
        wmh_coords = np.array(np.where(wmh_mask)).T

        wm_mask = (aseg_data == 41) | (aseg_data == 2)
        wm_coords = np.array(np.where(wm_mask)).T

        if wmh_coords.size > 0 and wm_coords.size > 0:
            for coord in wmh_coords:
                distances = np.linalg.norm(wm_coords - coord, axis=1)
                closest_wm_index = np.argmin(distances)
                closest_wm_coord = wm_coords[closest_wm_index]

                if aseg_data[tuple(closest_wm_coord)] == 41:
                    aseg_data[tuple(coord)] = 41
                else:
                    aseg_data[tuple(coord)] = 2

        # ----------------------------------------------
        # Step 2: Fix choroid plexus labels 31 and 63
        # using per-component contact with ventricles
        # vs. white matter
        # ----------------------------------------------

        # 6-connectivity
        struct = generate_binary_structure(3, 1)

        def count_contacts(label_mask, target_mask):
            # Dilate the component and count overlap with target
            dil = binary_dilation(label_mask, structure=struct)
            return int(np.sum(dil & target_mask))

        # ---- Left side: label 31 ----
        cp31_mask = aseg_data == 31
        if np.any(cp31_mask):
            labeled, num_labels = cc_label(cp31_mask, structure=struct)
            lv_mask = aseg_data == 4    # left ventricle
            lwm_mask = aseg_data == 2   # left white matter

            for lbl in range(1, num_labels + 1):
                comp_mask = labeled == lbl

                contact_lv = count_contacts(comp_mask, lv_mask)
                contact_lwm = count_contacts(comp_mask, lwm_mask)

                # If contact with LV is not greater than contact with WM,
                # treat this component as misclassified WM and relabel to 2.
                if contact_lv <= contact_lwm:
                    aseg_data[comp_mask] = 2

        # ---- Right side: label 63 ----
        cp63_mask = aseg_data == 63
        if np.any(cp63_mask):
            labeled, num_labels = cc_label(cp63_mask, structure=struct)
            rv_mask = aseg_data == 43   # right ventricle
            rwm_mask = aseg_data == 41  # right white matter

            for lbl in range(1, num_labels + 1):
                comp_mask = labeled == lbl

                contact_rv = count_contacts(comp_mask, rv_mask)
                contact_rwm = count_contacts(comp_mask, rwm_mask)

                # If contact with RV is not greater than contact with WM,
                # treat this component as misclassified WM and relabel to 41.
                if contact_rv <= contact_rwm:
                    aseg_data[comp_mask] = 41

        # -----------------------------
        # Save result
        # -----------------------------
        new_img = nib.Nifti1Image(aseg_data, aseg_img.affine, aseg_img.header)
        nib.save(new_img, out_aseg_path)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_aseg"] = self.inputs.out_aseg
        return outputs

# ============================================
# Connectome preparation
# ============================================

class ConnectomePrepareInputSpec(CommandLineInputSpec):
    preproc_dwi_mif = Str(
        mandatory=True,
        desc="Preprocessed DWI image in MRtrix .mif format (must contain gradient table)",
        argstr="%s",
        position=0,
    )
    dwi_mask_mif = Str(
        mandatory=True,
        desc="DWI brain mask in MRtrix .mif format",
        argstr="%s",
        position=1,
    )
    output_dir = Str(
        mandatory=True,
        desc="Output directory",
        argstr="%s",
        position=2,
    )
    aseg = Str(
        mandatory=True,
        desc="Asegmentation image in DWI space (NIfTI)",
        argstr="%s",
        position=3,
    )

    # Output filenames (relative names, will be joined with output_dir in bash)
    wm_response = Str(mandatory=True, desc="WM response function filename", argstr="%s", position=4)
    wm_fod = Str(mandatory=True, desc="WM FOD filename", argstr="%s", position=5)
    wm_fod_norm = Str(mandatory=True, desc="Normalized WM FOD filename", argstr="%s", position=6)

    gm_response = Str(mandatory=True, desc="GM response function filename", argstr="%s", position=7)
    gm_fod = Str(mandatory=True, desc="GM FOD filename", argstr="%s", position=8)
    gm_fod_norm = Str(mandatory=True, desc="Normalized GM FOD filename", argstr="%s", position=9)

    csf_response = Str(mandatory=True, desc="CSF response function filename", argstr="%s", position=10)
    csf_fod = Str(mandatory=True, desc="CSF FOD filename", argstr="%s", position=11)
    csf_fod_norm = Str(mandatory=True, desc="Normalized CSF FOD filename", argstr="%s", position=12)
    sift_mu = Str(mandatory=True, desc="SIFT2 mu output filename", argstr="%s", position=13)
    five_tt_dwi = Str(mandatory=True, desc="5TT in DWI space output filename", argstr="%s", position=14)
    gmwmSeed_dwi = Str(mandatory=True, desc="GMWMI seed in DWI space output filename", argstr="%s", position=15)
    streamlines = Str(mandatory=True, desc="Streamlines output filename", argstr="%s", position=16)
    sift_weights = Str(mandatory=True, desc="SIFT2 weights output filename", argstr="%s", position=17)

class ConnectomePrepareOutputSpec(TraitedSpec):
    sift_weights = File(desc="SIFT weights file")
    global_streamlines = File(desc="Streamlines file (global connectome)")


class ConnectomePrepare(CommandLine):
    _cmd = "bash " + get_package_path("pipelines", "bash", "mrtrix3", "mrtrix_connectome_preprocess.sh")
    input_spec = ConnectomePrepareInputSpec
    output_spec = ConnectomePrepareOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["sift_weights"] = os.path.join(self.inputs.output_dir, self.inputs.sift_weights)
        outputs["global_streamlines"] = os.path.join(self.inputs.output_dir, self.inputs.streamlines)
        return outputs