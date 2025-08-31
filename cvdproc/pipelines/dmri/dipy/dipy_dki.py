import os
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.data import fetch_hbn
import dipy.reconst.dti as dti
import dipy.reconst.fwdti as fwdti
import numpy as np
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
import sys, time, faulthandler, json, traceback
import numpy as np
import nibabel as nib
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dki import DiffusionKurtosisModel
from dipy.core.sphere import Sphere

class DKIFitInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True, desc="Path to the DWI file")
    bval_file = File(exists=True, mandatory=True, desc="Path to the bval file")
    bvec_file = File(exists=True, mandatory=True, desc="Path to the bvec file")
    mask_file = File(exists=True, mandatory=True, desc="Path to the brain mask file")
    output_dir = Directory(mandatory=True, desc="Output directory for the DKI metrics")

class DKIFitOutputSpec(TraitedSpec):
    fa_file = File(desc="Path to the output FA map")
    md_file = File(desc="Path to the output MD map")
    ad_file = File(desc="Path to the output AD map")
    rd_file = File(desc="Path to the output RD map")
    mk_file = File(desc="Path to the output MK map")
    ak_file = File(desc="Path to the output AK map")
    rk_file = File(desc="Path to the output RK map")
    kxxxx_file = File(desc="Path to the output Kxxxx map")
    kyyyy_file = File(desc="Path to the output Kyyyy map")
    kzzzz_file = File(desc="Path to the output Kzzzz map")
    diffusion_tensor_file = File(desc="Path to the output diffusion tensor map")
    kurtosis_tensor_file = File(desc="Path to the output kurtosis tensor map")

class DKIFit(BaseInterface):