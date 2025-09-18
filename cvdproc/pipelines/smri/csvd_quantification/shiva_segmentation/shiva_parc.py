#!/usr/bin/env python3
import os
import numpy as np
import json
import nibabel as nib
from functools import reduce
from scipy.ndimage import (
    distance_transform_edt,
    binary_dilation,
    binary_erosion,
    binary_closing,
    binary_opening
)
from scipy.spatial import Delaunay, ConvexHull

import os
from nipype.interfaces.base import (CommandLineInputSpec, CommandLine, traits, TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File)
from traits.api import Directory, Str, List, Bool, Enum, Int, Float, TraitError, Either

def lobar_and_wm_segmentation(
    seg_input,
    save_path=None,
    jxct_thickness=3,
    pvwm_thickness=2,
    ic_erosion_iter=2,
    cc_grow_iter=10,
    use_np_isin=True
):
    """
    Compute lobar + WM subdivisions (juxtacortical/deep/periventricular) and key WM tracts
    (internal capsule, external/extreme capsule, corpus callosum) from a FreeSurfer/SynthSeg-style
    segmentation volume.

    Parameters
    ----------
    seg_input : str or np.ndarray
        Either a path to a NIfTI file (.nii or .nii.gz) or a 3D numpy int array of labels.
    save_path : str or None
        If provided, save the resulting segmentation NIfTI to this path (.nii.gz recommended).
    jxct_thickness : int
        Iterations for cortex dilation to form juxtacortical WM band (in voxels).
    pvwm_thickness : int
        Iterations for periventricular WM dilation from ventricles (in voxels).
    ic_erosion_iter : int
        Erosion iterations on BG convex hull before intersecting with WM to get internal capsule.
    cc_grow_iter : int
        Dilation iterations to grow CC from seed.
    use_np_isin : bool
        If True, use np.isin for label membership checks; otherwise use custom reduce-OR comparison.

    Returns
    -------
    out_seg : np.ndarray (int16)
        The consolidated segmentation.
    affine, header : if seg_input is a path, returns the NIfTI affine and header for convenience.

    Notes
    -----
    - Assumes FreeSurfer-like label conventions:
        * Left WM = 2, Right WM = 41
        * Cortical parcels ~1000+ (left), ~2000+ (right)
        * Thalamus/BG/Cerebellum/Brainstem use common FS codes
    - If voxel spacing is highly anisotropic, geometric steps (e.g., convex hull) are done in index space.
      For strict geometry, consider resampling to isotropic prior to running.
    """
    # ----- helpers -----
    def fisin(arr, vals):
        if use_np_isin:
            return np.isin(arr, vals)
        try:
            arrl = [arr == v for v in vals]
        except TypeError:
            arrl = [arr == vals]
        return reduce(lambda x, y: x | y, arrl)

    def expand_label_masked(label_image, mask):
        nearest_idx = distance_transform_edt(
            label_image == 0, return_distances=False, return_indices=True
        )
        out = np.zeros(label_image.shape, dtype=np.int16)
        masked_idx = [dim_idx[mask] for dim_idx in nearest_idx]
        nearest_labels = label_image[tuple(masked_idx)]
        out[mask] = nearest_labels
        return out

    def safe_fill_hull(bool_vol):
        points = np.argwhere(bool_vol)
        if points.shape[0] < 4:
            # Not enough points to form a 3D hull; return original mask
            return bool_vol.astype(bool)
        try:
            hull = ConvexHull(points)
        except Exception:
            return bool_vol.astype(bool)
        try:
            deln = Delaunay(points[hull.vertices])
        except Exception:
            return bool_vol.astype(bool)
        idx = np.stack(np.indices(bool_vol.shape), axis=-1)
        inside = deln.find_simplex(idx) >= 0
        return inside

    def internal_caps(seg):
        bg_labels_L = [10, 11, 12, 13]   # thalamus + caudate + putamen + pallidum
        bg_labels_R = [49, 50, 51, 52]
        wm_L = seg == 2
        wm_R = seg == 41
        bg_L = fisin(seg, bg_labels_L)
        bg_R = fisin(seg, bg_labels_R)
        ic_L = binary_erosion(safe_fill_hull(bg_L), iterations=ic_erosion_iter) & wm_L
        ic_R = binary_erosion(safe_fill_hull(bg_R), iterations=ic_erosion_iter) & wm_R
        return ic_L, ic_R

    def ex_capsule(seg, exclusion_wm):
        wm_L = seg == 2
        wm_R = seg == 41
        hipp_vdc = fisin(seg, [17, 53, 28, 60])  # hippocampus L/R + ventral DC L/R
        hipp_vdc_dil = binary_dilation(hipp_vdc, iterations=5)
        exclusion = exclusion_wm | hipp_vdc_dil
        put_L, ins_L = (seg == 12), (seg == 1035)
        put_R, ins_R = (seg == 51), (seg == 2035)

        hull_L = safe_fill_hull(ins_L | put_L)
        hull_R = safe_fill_hull(ins_R | put_R)
        xcap_L_raw = hull_L & binary_dilation(put_L, iterations=10) & binary_dilation(ins_L, iterations=10)
        xcap_R_raw = hull_R & binary_dilation(put_R, iterations=10) & binary_dilation(ins_R, iterations=10)

        xcap_L = xcap_L_raw & wm_L & (~exclusion)
        xcap_R = xcap_R_raw & wm_R & (~exclusion)

        xcap_L = binary_closing(xcap_L | ins_L | put_L, iterations=3) & ~(ins_L | put_L) & wm_L
        xcap_R = binary_closing(xcap_R | ins_R | put_R, iterations=3) & ~(ins_R | put_R) & wm_R
        return xcap_L, xcap_R

    def juxtacortical_wm(seg, thickness=3):
        cortex_vals_L = []
        for _, vals in lobar_vals_L.items():
            cortex_vals_L += vals
        cortex_vals_R = []
        for _, vals in lobar_vals_R.items():
            cortex_vals_R += vals
        cortex = fisin(seg, cortex_vals_L + cortex_vals_R)
        cortex_dil = binary_dilation(cortex, iterations=thickness)
        return (cortex_dil & (seg == 2)), (cortex_dil & (seg == 41))

    def periventricular_wm(seg, thickness=2):
        wm_L = seg == 2
        wm_R = seg == 41
        vent_L = fisin(seg, [4, 5])
        vent_R = fisin(seg, [43, 44])
        pvwm_L = binary_dilation(vent_L, iterations=thickness) & wm_L
        pvwm_R = binary_dilation(vent_R, iterations=thickness) & wm_R
        return pvwm_L, pvwm_R

    def corpus_cal(seg):
        wm_L = seg == 2
        wm_R = seg == 41
        wm = (wm_L | wm_R).copy()

        vent_1_2 = fisin(seg, [4, 5, 43, 44])
        vent_3_dil6 = binary_dilation(seg == 14, iterations=6)
        vent_123 = binary_closing(vent_1_2 | vent_3_dil6, iterations=6)

        cing_L = binary_closing(fisin(seg, cingulate_vals_L), iterations=5)
        cing_R = binary_closing(fisin(seg, cingulate_vals_R), iterations=5)
        exclusion = vent_123 | cing_L | cing_R
        wm[exclusion] = False

        dil_wm_L = binary_dilation(wm_L) & wm
        dil_wm_R = binary_dilation(wm_R) & wm
        seed_cc = dil_wm_L & dil_wm_R

        cc_raw = binary_dilation(seed_cc, iterations=cc_grow_iter)
        cc_filtered = binary_opening(cc_raw & wm, iterations=2)
        return (cc_filtered & wm_L), (cc_filtered & wm_R)

    # ----- label dictionaries (as constants) -----
    lobar_vals_L = {
        'Frontal':   [1002, 1003, 1012, 1014, 1017, 1018, 1019, 1020, 1027, 1028, 1032, 1024, 1026],
        'Parietal':  [1008, 1010, 1025, 1029, 1031, 1022, 1023],
        'Temporal':  [1001, 1006, 1007, 1009, 1015, 1016, 1030, 1033, 1034],
        'Occipital': [1005, 1011, 1013, 1021],
    }
    lobar_vals_R = {k: [v + 1000 for v in val] for k, val in lobar_vals_L.items()}

    other_vals_L = {
        'Insula': [1035],
        'Basal Ganglia': [11, 12, 13, 26],
        'Thalamus': [10],
        'Ventral DC': [28],
        'Hippocampus': [17, 18],
        'Cerebellum': [7, 8],
    }
    other_vals_R = {
        'Insula': [2035],
        'Basal Ganglia': [50, 51, 52, 58],
        'Thalamus': [49],
        'Ventral DC': [60],
        'Hippocampus': [53, 54],
        'Cerebellum': [46, 47],
    }
    brainstem_val = 16

    to_excluded_L = [1002, 1017, 1026, 1010, 1023]
    to_excluded_R = [v + 1000 for v in to_excluded_L]

    cingulate_vals_L = [1002, 1010, 1023, 1026]
    cingulate_vals_R = [2002, 2010, 2023, 2026]

    lobar_labels_L = {
        'Frontal': 1,
        'Parietal': 5,
        'Temporal': 9,
        'Occipital': 13,
    }
    lobar_labels_R = {k: v + 20 for k, v in lobar_labels_L.items()}

    other_labels_L = {
        'Insula': 17,
        'Basal Ganglia': 40,
        'Thalamus': 41,
        'Ventral DC': 42,
        'Hippocampus': 43,
        'Internal capsule': 44,
        'Ext. capsule': 45,
        'Corpus callosum': 46,
        'Cerebellum': 47,
    }
    other_labels_R = {k: v + 10 for k, v in other_labels_L.items()}
    other_labels_R['Insula'] = other_labels_L['Insula'] + 20
    brainstem_label = 60

    # ----- load input -----
    if isinstance(seg_input, str):
        seg_nii = nib.load(seg_input)
        seg = seg_nii.get_fdata().astype(np.int16)
        affine, header = seg_nii.affine, seg_nii.header
    else:
        seg = np.asarray(seg_input, dtype=np.int16)
        affine, header = None, None

    # ----- step 1: initial lobar GM labels (per hemisphere) -----
    wm_L = seg == 2
    wm_R = seg == 41

    # Treat excluded cortical parcels as WM seeds to avoid "jigsaw" artifacts
    wm_L = wm_L | fisin(seg, to_excluded_L)
    wm_R = wm_R | fisin(seg, to_excluded_R)

    vol_lobar_L = np.zeros(seg.shape, dtype=np.int16)
    vol_lobar_R = np.zeros(seg.shape, dtype=np.int16)

    for lob in lobar_vals_L.keys():
        vals_L = list(set(lobar_vals_L[lob]) - set(to_excluded_L))
        vals_R = list(set(lobar_vals_R[lob]) - set(to_excluded_R))
        vol_lobar_L[fisin(seg, vals_L)] = lobar_labels_L[lob]
        vol_lobar_R[fisin(seg, vals_R)] = lobar_labels_R[lob]

    # ----- step 2: expand GM labels into WM (assign WM to nearest lobe) -----
    vol_lobar_L_exp = expand_label_masked(vol_lobar_L, wm_L) + vol_lobar_L
    vol_lobar_R_exp = expand_label_masked(vol_lobar_R, wm_R) + vol_lobar_R

    # ----- step 3: add other structures with compact labels -----
    for parc in other_vals_L:
        vol_lobar_L_exp[fisin(seg, other_vals_L[parc])] = other_labels_L[parc]
    for parc in other_vals_R:
        vol_lobar_R_exp[fisin(seg, other_vals_R[parc])] = other_labels_R[parc]

    out_seg = vol_lobar_L_exp + vol_lobar_R_exp
    out_seg[seg == brainstem_val] = brainstem_label

    # ----- step 4: compute WM subclasses and key tracts -----
    ic_L, ic_R = internal_caps(seg)
    jxtc_L, jxtc_R = juxtacortical_wm(seg, thickness=jxct_thickness)
    prvc_L, prvc_R = periventricular_wm(seg, thickness=pvwm_thickness)
    ec_L, ec_R = ex_capsule(seg, (jxtc_L | jxtc_R | ic_L | ic_R))
    cc_L, cc_R = corpus_cal(seg)

    # Re-label lobe WM into juxtacortical / deep / periventricular
    for _, base_label in lobar_labels_L.items():
        lobar_mask = (out_seg == base_label) & (seg == 2)
        jx = lobar_mask & jxtc_L
        pv = lobar_mask & prvc_L
        out_seg[lobar_mask] = base_label + 2   # deep WM by default
        out_seg[jx] = base_label + 1           # jxct WM
        out_seg[pv] = base_label + 3           # periventricular WM

    for _, base_label in lobar_labels_R.items():
        lobar_mask = (out_seg == base_label) & (seg == 41)
        jx = lobar_mask & jxtc_R
        pv = lobar_mask & prvc_R
        out_seg[lobar_mask] = base_label + 2
        out_seg[jx] = base_label + 1
        out_seg[pv] = base_label + 3

    # Overwrite key tracts
    out_seg[ic_L] = other_labels_L['Internal capsule']
    out_seg[ic_R] = other_labels_R['Internal capsule']
    out_seg[ec_L] = other_labels_L['Ext. capsule']
    out_seg[ec_R] = other_labels_R['Ext. capsule']
    out_seg[cc_L] = other_labels_L['Corpus callosum']
    out_seg[cc_R] = other_labels_R['Corpus callosum']

    out_seg = out_seg.astype(np.int16)

    # ----- save if requested -----
    if save_path is not None:
        if affine is None or header is None:
            raise ValueError("To save NIfTI, seg_input must be a path so affine/header are available.")
        nib.save(nib.Nifti1Image(out_seg, affine, header), save_path)

    if isinstance(seg_input, str):
        return out_seg, affine, header
    return out_seg, None, None

class ShivaGeneralParcellationInputSpec(BaseInterfaceInputSpec):
    in_seg = File(exists=True, mandatory=True, desc="Input segmentation NIfTI file (FreeSurfer/SynthSeg style)")
    out_seg = Str(mandatory=False, desc="Output parcellation NIfTI file path")
    jxct_thickness = Int(3, usedefault=True, desc="Iterations for cortex dilation to form juxtacortical WM band (in voxels)")
    pvwm_thickness = Int(2, usedefault=True, desc="Iterations for periventricular WM dilation from ventricles (in voxels)")
    ic_erosion_iter = Int(2, usedefault=True, desc="Erosion iterations on BG convex hull before intersecting with WM to get internal capsule")
    cc_grow_iter = Int(10, usedefault=True, desc="Dilation iterations to grow CC from seed")
    use_np_isin = Bool(True, usedefault=True, desc="If True, use np.isin for label membership checks; otherwise use custom reduce-OR comparison")
class ShivaGeneralParcellationOutputSpec(TraitedSpec):
    out_seg = File(desc="Output parcellation NIfTI file path")
class ShivaGeneralParcellation(BaseInterface):
    input_spec = ShivaGeneralParcellationInputSpec
    output_spec = ShivaGeneralParcellationOutputSpec

    def _run_interface(self, runtime):
        in_seg = self.inputs.in_seg
        out_seg = self.inputs.out_seg
        jxct_thickness = self.inputs.jxct_thickness
        pvwm_thickness = self.inputs.pvwm_thickness
        ic_erosion_iter = self.inputs.ic_erosion_iter
        cc_grow_iter = self.inputs.cc_grow_iter
        use_np_isin = self.inputs.use_np_isin

        if out_seg is None or out_seg.strip() == "":
            out_dir = os.path.dirname(in_seg)
            out_fname = os.path.basename(in_seg).replace(".nii.gz", "_shiva_parc.nii.gz").replace(".nii", "_shiva_parc.nii")
            out_seg = os.path.join(out_dir, out_fname)

        lobar_and_wm_segmentation(
            in_seg,
            save_path=out_seg,
            jxct_thickness=jxct_thickness,
            pvwm_thickness=pvwm_thickness,
            ic_erosion_iter=ic_erosion_iter,
            cc_grow_iter=cc_grow_iter,
            use_np_isin=use_np_isin
        )

        self._out_file = out_seg
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_seg"] = self._out_file
        return outputs

'''
Custom parcellation functions for each cSVD biomarker
Each function requires our Synthseg-derived parcellation
'''

import numpy as np
from typing import Tuple

def fisin(arr, vals):
    '''
    Fast np.isin function using reccursive bitwise_or function 
    (here represented with the lambda function, because slightly faster(?))
    '''
    try:
        arrl = [arr == val for val in vals]
    except TypeError:  # Typically if there is only 1 value
        arrl = [arr == vals]
    return reduce(lambda x, y: x | y, arrl)

def seg_for_pvs(parc: np.ndarray) -> Tuple[np.ndarray, dict]:
    '''
    Will segment PVS into:  L / R
        - Deep:             1 / 6
        - Basal Ganglia:    2 / 7
        - Hippocampal:      3 / 8
        - Cerebellar:       4 / 9
        - Ventral DC:       5 / 10
        - Brainstem:          11

    '''
    seg_vals = {  # See src/shivai/postprocessing/lobarseg.py for labels
        # Left
        'Left Deep WM': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 46],  # With cortical GM for now
        'Left Basal Gang.': [17, 40, 41, 44, 45],
        'Left Hippoc.': [43],
        'Left Cerebellar': [47],
        'Left Ventral DC': [42],
        # Right
        'Right Deep WM': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 56],
        'Right Basal Gang.': [37, 50, 51, 54, 55],
        'Right Hippoc.': [53],
        'Right Cerebellar': [57],
        'Right Ventral DC': [52],
        #
        'Brainstem': 60
    }
    seg_labels = {
        # Left
        'Left Deep WM': 1,
        'Left Basal Gang.': 2,
        'Left Hippoc.': 3,
        'Left Cerebellar': 4,
        'Left Ventral DC': 5,
        # Right
        'Right Deep WM': 6,
        'Right Basal Gang.': 7,
        'Right Hippoc.': 8,
        'Right Cerebellar': 9,
        'Right Ventral DC': 10,
        #
        'Brainstem': 11
    }

    pvs_seg = np.zeros(parc.shape, 'int16')
    for region, vals in seg_vals.items():
        pvs_seg[fisin(parc, vals)] = seg_labels[region]

    return pvs_seg, seg_labels


def seg_for_wmh(parc: np.ndarray) -> Tuple[np.ndarray, dict]:
    '''
    Will segment WMH into:  L / R
        - Shallow:          1 / 5  (includes the cortex, i.e. GM)
        - Deep:             2 / 6
        - Perivetricular:   3 / 7
        - Cerebellar:       4 / 8
        - Brainstem:          9
    '''
    seg_vals = {  # See src/shivai/postprocessing/lobarseg.py for labels
        # Left
        'Left Shallow WM': [1, 2, 5, 6, 9, 10, 13, 14, 17, 45],  # With cortical GM
        'Left Deep WM': [3, 7, 11, 15, 44, 46],
        'Left PV WM': [4, 8, 12, 16],
        'Left Cerebellar': [47],
        # Right
        'Right Shallow WM': [21, 22, 25, 26, 29, 30, 33, 34, 37, 55],  # With cortical GM
        'Right Deep WM': [23, 27, 31, 35, 54, 56],
        'Right PV WM': [24, 28, 32, 36],
        'Right Cerebellar': [57],
        #
        'Brainstem': 60
    }
    seg_labels = {
        # Left
        'Left Shallow WM': 1,  # With cortical GM
        'Left Deep WM': 2,
        'Left PV WM': 3,
        'Left Cerebellar': 4,
        # Right
        'Right Shallow WM': 5,  # With cortical GM
        'Right Deep WM': 6,
        'Right PV WM': 7,
        'Right Cerebellar': 8,
        #
        'Brainstem': 9
    }
    wmh_seg = np.zeros(parc.shape, 'int16')
    for region, vals in seg_vals.items():
        wmh_seg[fisin(parc, vals)] = seg_labels[region]
    return wmh_seg, seg_labels


def seg_from_mars(parc: np.ndarray) -> Tuple[np.ndarray, dict]:
    '''
    Based on the Microbleed Anatomical Rating Scale (MARS)
    Will segment CMBs into: L / R
        - Frontal:          1 / 15
        - Parietal:         2 / 16
        - Temporal:         3 / 17
        - Occipital:        4 / 18
        - Insula:           5 / 19 (just the GM)
        - Basal Ganglia:    6 / 20
        - Thalamus:         7 / 21
        - Ventral DC:       8 / 22
        - Hippoc.:      9 / 23
        - Int. Caps.:        10 / 24
        - Ext. Caps.:        11 / 25
        - Corp. Call.:      12 / 26
        - D&PV WM:          13 / 27
        - Cerebellum:       14 / 28
        - Brainstem:          29

    '''
    seg_vals = {  # See src/shivai/postprocessing/lobarseg.py for labels
        # Left
        'Left Frontal': [1, 2],
        'Left Parietal': [5, 6],
        'Left Temporal': [9, 10],
        'Left Occipital': [13, 14],
        'Left Insula': [17],
        'Left Basal Gang.': [40],
        'Left Thalamus': [41],
        'Left Ventral DC': [42],
        'Left Hippoc.': [43],
        'Left Int. Caps.': [44],
        'Left Ext. Caps.': [45],
        'Left Corp. Call.': [46],
        'Left D&PV WM': [3, 4, 7, 8, 11, 12, 15, 16],
        'Left Cerebellum': [47],
        # Right
        'Right Frontal': [21, 22],
        'Right Parietal': [25, 26],
        'Right Temporal': [29, 30],
        'Right Occipital': [33, 34],
        'Right Insula': [37],
        'Right Basal Gang.': [50],
        'Right Thalamus': [51],
        'Right Ventral DC': [52],
        'Right Hippoc.': [53],
        'Right Int. Caps.': [54],
        'Right Ext. Caps.': [55],
        'Right Corp. Call.': [56],
        'Right D&PV WM': [23, 24, 27, 28, 31, 32, 35, 36],
        'Right Cerebellum': [57],
        #
        'Brainstem': 60
    }
    seg_labels = {
        # Left
        'Left Frontal': 1,
        'Left Parietal': 2,
        'Left Temporal': 3,
        'Left Occipital': 4,
        'Left Insula': 5,
        'Left Basal Gang.': 6,
        'Left Thalamus': 7,
        'Left Ventral DC': 8,
        'Left Hippoc.': 9,
        'Left Int. Caps.': 10,
        'Left Ext. Caps.': 11,
        'Left Corp. Call.': 12,
        'Left D&PV WM': 13,
        'Left Cerebellum': 14,
        # Right
        'Right Frontal': 15,
        'Right Parietal': 16,
        'Right Temporal': 17,
        'Right Occipital': 18,
        'Right Insula': 19,
        'Right Basal Gang.': 20,
        'Right Thalamus': 21,
        'Right Ventral DC': 22,
        'Right Hippoc.': 23,
        'Right Int. Caps.': 24,
        'Right Ext. Caps.': 25,
        'Right Corp. Call.': 26,
        'Right D&PV WM': 27,
        'Right Cerebellum': 28,
        #
        'Brainstem': 29
    }
    cmb_seg = np.zeros(parc.shape, 'int16')
    for region, vals in seg_vals.items():
        cmb_seg[fisin(parc, vals)] = seg_labels[region]
    return cmb_seg, seg_labels

class Brain_Seg_for_biomarker_InputSpec(BaseInterfaceInputSpec):

    brain_seg = traits.File(exists=True,
                            mandatory=True,
                            desc='"derived_parc" brain segmentation from "Parc_from_Synthseg" node.')

    custom_parc = traits.Str('pvs',
                             usedefault=True,
                             desc='Type of custom parcellisation scheme to use. Can be "pvs", "wmh, or "mars"')

    out_file = traits.Str('Brain_Seg_for_biomarker.nii.gz',
                          usedefault=True,
                          desc='Filename of the ouput segmentation')
    parc_json = traits.Str('parc_dict.json',
                          usedefault=True,
                          desc='Filename of the json file where the region_dict will be saved for future reference')


class Brain_Seg_for_biomarker_OutputSpec(TraitedSpec):
    brain_seg = traits.File(exists=True,
                            desc='Brain segmentation, derived from synthseg segmentation, and used for the biomarker metrics')
    region_dict = traits.Dict(key_trait=traits.Str,
                              value_trait=traits.Int,
                              desc=('Dictionnary with keys = brain region names, '
                                    'and values = brain region labels (i.e. the corresponding value in brain_seg)'))
    region_dict_json = traits.File(exists=True,
                                   desc='json file where region_dict will be saved for future reference')


class Brain_Seg_for_biomarker(BaseInterface):
    """
    Transform our parcellation (derived from Synthseg) to one that is customized for the studied biomarker metrics.
    'mars' refers to the Microbleed Anatomical Rating Scale (MARS)
    """
    input_spec = Brain_Seg_for_biomarker_InputSpec
    output_spec = Brain_Seg_for_biomarker_OutputSpec

    def _run_interface(self, runtime):
        seg_im = nib.load(self.inputs.brain_seg)
        seg_vol = seg_im.get_fdata().astype(int)

        seg_scheme = self.inputs.custom_parc
        if seg_scheme == 'pvs':
            custom_seg_vol, seg_dict = seg_for_pvs(seg_vol)
        elif seg_scheme == 'wmh':
            custom_seg_vol, seg_dict = seg_for_wmh(seg_vol)
        elif seg_scheme == 'mars':
            custom_seg_vol, seg_dict = seg_from_mars(seg_vol)
        else:
            raise ValueError(f'Unrecognised segmentation scheme: Expected "pvs", "wmh", or "mars" but bot "{seg_scheme}"')

        region_dict = {'Whole brain': -1, **seg_dict}
        custom_seg_im = nib.Nifti1Image(custom_seg_vol, affine=seg_im.affine)
        nib.save(custom_seg_im, self.inputs.out_file)

        setattr(self, 'region_dict', region_dict)
        json_name = self.inputs.parc_json
        setattr(self, 'json_name', json_name)
        with open(json_name, 'w') as jsonfile:
            json.dump(region_dict, jsonfile, indent=4)

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['brain_seg'] = os.path.abspath(self.inputs.out_file)
        outputs['region_dict'] = getattr(self, 'region_dict')
        outputs['region_dict_json'] = os.path.abspath(self.inputs.parc_json)

        return outputs

if __name__ == "__main__":
    csvd_parc = Brain_Seg_for_biomarker()
    csvd_parc.inputs.brain_seg = '/mnt/f/BIDS/demo_BIDS/derivatives/anat_seg/sub-TAOHC0261/ses-baseline/synthseg/sub-TAOHC0261_ses-baseline_space-T1w_shiva_parc.nii.gz'
    csvd_parc.inputs.custom_parc = 'pvs'
    csvd_parc.inputs.out_file = '/mnt/f/BIDS/demo_BIDS/derivatives/anat_seg/sub-TAOHC0261/ses-baseline/synthseg/test_pvs_seg.nii.gz'
    csvd_parc.inputs.parc_json = '/mnt/f/BIDS/demo_BIDS/derivatives/anat_seg/sub-TAOHC0261/ses-baseline/synthseg/test_pvs_seg.json'
    res = csvd_parc.run()