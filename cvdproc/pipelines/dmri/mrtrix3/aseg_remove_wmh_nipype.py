import os
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, Str
)
import nibabel as nib
import numpy as np
from scipy.ndimage import (
    generate_binary_structure,
    binary_dilation,
    label as cc_label,
)


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


if __name__ == "__main__":
    # Example usage
    remover = RemoveWMH()
    remover.inputs.in_aseg = '/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/dwi_pipeline/sub-SSI0248/ses-baseline/aparc_aseg.nii.gz'
    remover.inputs.out_aseg = '/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/dwi_pipeline/sub-SSI0248/ses-baseline/aparc_aseg_no_wmh.nii.gz'
    remover.run()