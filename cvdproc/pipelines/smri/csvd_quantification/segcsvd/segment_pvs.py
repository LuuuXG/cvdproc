import os
from nipype.interfaces.base import (CommandLineInputSpec, CommandLine, traits, TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File)
from traits.api import Directory, Str, List, Bool, Enum, Int, Float, TraitError, Either

class SegCSVDPVSInputSpec(CommandLineInputSpec):
    t1w = Str(mandatory=True, argstr='%s', position=0, desc='T1-weighted image (skull-stripped and bias-corrected)')
    synthseg_out = Str(mandatory=True, argstr='%s', position=1, desc='SynthSeg output segmentation file')
    wmh_file = Str(argstr='%s', position=2, desc='Optional: WMH segmentation file; if not provided, a pseudo WMH (all=0) will be created')
    output_dir = Directory(mandatory=True, argstr='%s', position=3, desc='Output directory for PVS segmentation results')
    pvs_probmap_filename = Str(argstr='%s', position=4, desc='Filename of the PVS probability map file')
    pvs_binary_filename = Str(argstr='%s', position=5, desc='Filename of the PVS binary segmentation file')
    threshold = Float(0.35, argstr='%f', position=6, desc='Threshold for binary PVS map (default: 0.35)')

class SegCSVDPVSOutputSpec(TraitedSpec):
    pvs_probmap = File(desc='PVS probability map output file')
    pvs_binary = File(desc='PVS binary segmentation output file')

class SegCSVDPVS(CommandLine):
    """
    Nipype interface for CSVD PVS segmentation command.
    """
    _cmd = 'bash ' + os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'bash', 'segcsvd', 'segcsvd_pvs.sh'))
    input_spec = SegCSVDPVSInputSpec
    output_spec = SegCSVDPVSOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['pvs_probmap'] = os.path.abspath(os.path.join(self.inputs.output_dir, self.inputs.pvs_probmap_filename))
        outputs['pvs_binary'] = os.path.abspath(os.path.join(self.inputs.output_dir, self.inputs.pvs_binary_filename))
        return outputs