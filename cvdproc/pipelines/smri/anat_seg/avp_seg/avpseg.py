import os
from nipype.interfaces.base import (TraitedSpec, CommandLineInputSpec, CommandLine, File, Str, Directory)
from traits.api import Bool

class AvpSegInputSpec(CommandLineInputSpec):
    t1w = File(exists=True, mandatory=True, desc='Input T1-weighted image', argstr='%s', position=0)
    synthseg = File(exists=True, mandatory=True, desc='SynthSeg segmentation in T1w space', argstr='%s', position=1)
    output_dir = Directory(mandatory=True, desc='Output directory', argstr='%s', position=2)
    mask_name = Str(argstr='--mask-name %s', desc='AVP mask file name')
    prob_name = Str(argstr='--prob-name %s', desc='AVP probability file name')
    five_label_name = Str(argstr='--five-label-name %s', desc='5-label segmentation file name')
    metrics_name = Str(argstr='--metrics-name %s', desc='Optic nerve metrics file name')
    qc_name = Str(argstr='--qc-name %s', desc='QC HTML file name')
    overwrite = Bool(argstr='--overwrite', desc='Replace existing outputs')
    keep_work = Bool(argstr='--keep-work', desc='Keep intermediate work directory')

class AvpSegOutputSpec(TraitedSpec):
    out_mask = File(desc='AVP mask file')
    out_prob = File(desc='AVP probability file')
    out_5label = File(desc='5-label segmentation file')
    out_metrics = File(desc='Optic nerve metrics Excel file')
    out_qc = File(desc='QC HTML file')

class AvpSeg(CommandLine):
    """Nipype interface for anterior visual pathway segmentation using avp_seg."""
    input_spec = AvpSegInputSpec
    output_spec = AvpSegOutputSpec
    _cmd = 'avp_seg'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        out_dir = self.inputs.output_dir
        mask_name = self.inputs.mask_name or 'avp_mask.nii.gz'
        prob_name = self.inputs.prob_name or 'avp_prob.nii.gz'
        five_label_name = self.inputs.five_label_name or 'avp_5label.nii.gz'
        metrics_name = self.inputs.metrics_name or 'optic_nerve_metrics.xlsx'
        qc_name = self.inputs.qc_name or 'qc.html'
        outputs['out_mask'] = os.path.join(out_dir, mask_name)
        outputs['out_prob'] = os.path.join(out_dir, prob_name)
        outputs['out_5label'] = os.path.join(out_dir, five_label_name)
        outputs['out_metrics'] = os.path.join(out_dir, metrics_name)
        outputs['out_qc'] = os.path.join(out_dir, qc_name)
        return outputs
