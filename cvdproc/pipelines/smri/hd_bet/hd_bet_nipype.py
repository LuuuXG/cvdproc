import os
from nipype.interfaces.base import (CommandLineInputSpec, CommandLine, traits, TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File)
from traits.api import Directory, Str, List, Bool, Enum, Int, Float, TraitError, Either
from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import isdefined


from nipype.interfaces.base import (
    Undefined,
)

from cvdproc.config.paths import get_package_path

# usage: hd-bet [-h] -i INPUT [-o OUTPUT] [-mode MODE] [-device DEVICE] [-tta TTA] [-pp PP] [-s SAVE_MASK]
#               [--overwrite_existing OVERWRITE_EXISTING] [-b BET]

# optional arguments:
#   -h, --help            show this help message and exit
#   -i INPUT, --input INPUT
#                         input. Can be either a single file name or an input folder. If file: must be nifti (.nii.gz)
#                         and can only be 3D. No support for 4d images, use fslsplit to split 4d sequences into 3d
#                         images. If folder: all files ending with .nii.gz within that folder will be brain extracted.
#   -o OUTPUT, --output OUTPUT
#                         output. Can be either a filename or a folder. If it does not exist, the folder will be created
#   -mode MODE            can be either 'fast' or 'accurate'. Fast will use only one set of parameters whereas accurate
#                         will use the five sets of parameters that resulted from our cross-validation as an ensemble.
#                         Default: accurate
#   -device DEVICE        used to set on which device the prediction will run. Must be either int or str. Use int for
#                         GPU id or 'cpu' to run on CPU. When using CPU you should consider disabling tta. Default for
#                         -device is: 0
#   -tta TTA              whether to use test time data augmentation (mirroring). 1= True, 0=False. Disable this if you
#                         are using CPU to speed things up! Default: 1
#   -pp PP                set to 0 to disabe postprocessing (remove all but the largest connected component in the
#                         prediction. Default: 1
#   -s SAVE_MASK, --save_mask SAVE_MASK
#                         if set to 0 the segmentation mask will not be saved
#   --overwrite_existing OVERWRITE_EXISTING
#                         set this to 0 if you don't want to overwrite existing predictions
#   -b BET, --bet BET     set this to 0 if you don't want to save skull-stripped brain

# Here we only accepted single file i/o
def _derive_mask_path_from_output(output_path: str) -> str:
    """
    Derive mask path by replacing output suffix:
      - *.nii.gz -> *_mask.nii.gz
      - *.nii    -> *_mask.nii
    """
    out_abs = os.path.abspath(output_path)

    if out_abs.endswith(".nii.gz"):
        return out_abs[:-7] + "_mask.nii.gz"
    if out_abs.endswith(".nii"):
        return out_abs[:-4] + "_mask.nii"

    raise ValueError(
        f"Output must end with .nii or .nii.gz for single-file I/O, got: {output_path}"
    )


class HDBetInputSpec(CommandLineInputSpec):
    input = File(
        exists=True,
        desc="Input NIfTI file (3D, .nii.gz/.nii).",
        argstr="-i %s",
        mandatory=True,
    )

    output = File(
        desc="Output skull-stripped brain image path (NIfTI .nii/.nii.gz).",
        argstr="-o %s",
        mandatory=True,
    )

    mode = Enum(
        "accurate",
        "fast",
        usedefault=True,
        default_value="accurate",
        desc="Mode: accurate|fast.",
        argstr="-mode %s",
    )

    device = Either(
        Int(),
        Str(),
        usedefault=True,
        default_value=0,
        desc="Device: GPU id (int) or 'cpu'.",
        argstr="-device %s",
    )

    tta = Int(
        usedefault=True,
        default_value=1,
        desc="Test-time augmentation: 1=True, 0=False.",
        argstr="-tta %d",
    )

    pp = Int(
        usedefault=True,
        default_value=1,
        desc="Postprocessing: 1=True, 0=False.",
        argstr="-pp %d",
    )

    save_mask = Int(
        usedefault=True,
        default_value=1,
        desc="Save mask: 1=True, 0=False.",
        argstr="-s %d",
    )

    overwrite_existing = Int(
        usedefault=True,
        default_value=1,
        desc="Overwrite existing outputs: 1=True, 0=False.",
        argstr="--overwrite_existing %d",
    )

    bet = Int(
        usedefault=True,
        default_value=1,
        desc="Save skull-stripped brain: 1=True, 0=False.",
        argstr="-b %d",
    )


class HDBetOutputSpec(TraitedSpec):
    output_brain = traits.Either(
        File(),
        None,
        desc="Output skull-stripped brain image (if bet==1).",
    )
    output_mask = traits.Either(
        File(),
        None,
        desc="Output brain mask (if save_mask==1).",
    )


class HDBet(CommandLine):
    """
    Nipype interface for hd-bet (single-file I/O convention).

    Output conventions:
      - output_brain: exactly the path provided by -o (if bet==1)
      - output_mask : derived from -o by suffix replacement (if save_mask==1)
    """

    input_spec = HDBetInputSpec
    output_spec = HDBetOutputSpec

    def __init__(self, **inputs):
        super().__init__(**inputs)
        self._cmd = self._resolve_cmd()

    @staticmethod
    def _resolve_cmd() -> str:
        """
        Resolve hd-bet executable path within your package, or fallback to 'hd-bet'
        if you prefer system PATH.
        """
        try:
            cmd = get_package_path("data", "lqt", "extdata", "HD_BET", "hd-bet")
            return cmd
        except Exception:
            return "hd-bet"

    def _validate_output_path(self, output_path: str) -> None:
        out_abs = os.path.abspath(output_path)
        if not (out_abs.endswith(".nii.gz") or out_abs.endswith(".nii")):
            raise TraitError(
                f"-o/--output must be a NIfTI file ending with .nii or .nii.gz, got: {output_path}"
            )

        out_dir = os.path.dirname(out_abs)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

    def _format_arg(self, name, spec, value):
        if name == "output" and isdefined(value):
            self._validate_output_path(value)
            return spec.argstr % os.path.abspath(value)

        if name == "input" and isdefined(value):
            return spec.argstr % os.path.abspath(value)

        return super()._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()

        if not isdefined(self.inputs.output):
            outputs["output_brain"] = None
            outputs["output_mask"] = None
            return outputs

        out_abs = os.path.abspath(self.inputs.output)

        bet_on = True
        if isdefined(self.inputs.bet):
            bet_on = int(self.inputs.bet) == 1

        mask_on = True
        if isdefined(self.inputs.save_mask):
            mask_on = int(self.inputs.save_mask) == 1

        outputs["output_brain"] = out_abs if bet_on else None
        outputs["output_mask"] = _derive_mask_path_from_output(out_abs) if mask_on else None

        return outputs