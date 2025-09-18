import os
from nipype.interfaces.base import (CommandLineInputSpec, CommandLine, traits, TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File)
from traits.api import Directory, Str, List, Bool, Enum, Int, Float, TraitError, Either

# ***************************************************
# The following COMPULSORY options have not been set:
#         -i,--in primary image filename (calc distance to positive voxels)
#         -o,--out        output image filename
# ***************************************************

# Part of FSL (ID: "")
# distancemap (Version 3.0)
# Copyright(c) 2003-2022, University of Oxford

# Usage:
# distancemap [options] -i <inputimage> -o <outputimage>

# Compulsory arguments (You MUST set one or more of):
#         -i,--in primary image filename (calc distance to positive voxels)
#         -o,--out        output image filename

# Optional arguments (You may optionally specify one or more of):
#         --secondim      second image filename (calc closest distance of this and primary input image, using positive voxels, negative distances mean this secondary image is the closer one)
#         -l,--localmax   local maxima output image filename
#         --invert        invert input image
#         --interp        filename for values to interpolate, valid samples given by primary image
#         --nn    nearest sparse interpolation
#         -v,--verbose    switch on diagnostic messages
#         -h,--help       display this message

class DistanceMapInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='-i %s', position=0, desc='Primary image filename (calc distance to positive voxels)')
    out_file = Str(mandatory=True, argstr='-o %s', position=1, desc='Output image filename')
    second_image = File(exists=True, argstr='--secondim %s', desc='Second image filename (calc closest distance of this and primary input image, using positive voxels, negative distances mean this secondary image is the closer one)')
    localmax = Str(argstr='-l %s', desc='Local maxima output image filename')
    invert = Bool(argstr='--invert', desc='Invert input image')
    interp = File(exists=True, argstr='--interp %s', desc='Filename for values to interpolate, valid samples given by primary image')
    nn = Bool(argstr='--nn', desc='Nearest sparse interpolation')
    verbose = Bool(argstr='-v', desc='Switch on diagnostic messages')

class DistanceMapOutputSpec(TraitedSpec):
    out_file = File(desc='Output distance map image file')
    localmax = Either(File, None, desc='Local maxima output image file')

class DistanceMap(CommandLine):
    """
    Nipype interface for FSL distancemap command.
    """
    _cmd = 'distancemap'
    input_spec = DistanceMapInputSpec
    output_spec = DistanceMapOutputSpec
    #terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        outputs['localmax'] = os.path.abspath(self.inputs.localmax) if self.inputs.localmax else None
        return outputs