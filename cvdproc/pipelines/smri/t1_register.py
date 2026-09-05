import os
from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface
from cvdproc.pipelines.common.register import SynthmorphNonlinear

from ...bids_data.rename_bids_file import rename_bids_file


# Template definitions: space -> {resolution: (filename, register_between_stripped)}
# For MNIPediatricAsym, filename uses {cohort} placeholder.
_TEMPLATES = {
    "MNI152NLin6Asym": {
        1:    ("MNI152_T1_1mm_brain.nii.gz", True),
        0.5:  ("MNI152_T1_0.5mm.nii.gz", False),
    },
    "MNI152NLin2009cAsym": {
        1:    ("tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz", False),
    },
    "MNIPediatricAsym": {
        1:    ("tpl-MNIPediatricAsym_cohort-{cohort}_res-1_T1w.nii.gz", False),
    },
}

# Spaces that require a cohort parameter
_COHORT_SPACES = {"MNIPediatricAsym"}


class T1RegisterPipeline:
    def __init__(self, subject, session, output_path, use_which_t1w: str = None,
                 template_space: str = "MNI152NLin6Asym", cohort: str = "", resolution: float = 1, **kwargs):
        """
        T1w registration pipeline to register T1w images to MNI space using SynthMorph.

        Args:
            subject (BIDSSubject): A BIDS subject object.
            session (BIDSSession): A BIDS session object.
            output_path (str): Output directory to save results.
            use_which_t1w (str, optional): Keyword to select the desired T1w image.
            template_space (str): Target MNI template space.
                Supported: "MNI152NLin6Asym" (default), "MNI152NLin2009cAsym", "MNIPediatricAsym".
            cohort (str): Pediatric cohort number. Only used with MNIPediatricAsym. Default "".
            resolution (float): Template resolution in mm. Default 1.
                MNI152NLin6Asym: 1 or 0.5.
                MNI152NLin2009cAsym: 1.
                MNIPediatricAsym: 1.
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.use_which_t1w = use_which_t1w
        self.template_space = template_space
        self.cohort = cohort
        self.resolution = resolution

        if self.template_space not in _TEMPLATES:
            raise ValueError(f"Unsupported template_space '{self.template_space}'. "
                             f"Choose from: {list(_TEMPLATES.keys())}.")
        if self.resolution not in _TEMPLATES[self.template_space]:
            raise ValueError(f"Resolution {self.resolution}mm is not available for {self.template_space}. "
                             f"Available: {list(_TEMPLATES[self.template_space].keys())}.")
        if self.template_space in _COHORT_SPACES and not self.cohort:
            raise ValueError(f"cohort is required for template_space '{self.template_space}'.")
        if self.template_space not in _COHORT_SPACES and self.cohort:
            raise ValueError(f"cohort is not applicable for template_space '{self.template_space}'.")

    def _get_template_filename(self):
        """Return (filename, register_between_stripped) for the current config."""
        filename_tpl, stripped = _TEMPLATES[self.template_space][self.resolution]
        if self.template_space in _COHORT_SPACES:
            filename_tpl = filename_tpl.format(cohort=self.cohort)
        return filename_tpl, stripped

    def check_data_requirements(self):
        return self.session.get_t1w_files() is not None

    def create_workflow(self):
        # get T1w image
        t1w_files = self.session.get_t1w_files()

        if self.use_which_t1w:
            t1w_files = [f for f in t1w_files if self.use_which_t1w in f]
            if len(t1w_files) != 1:
                raise FileNotFoundError(f"No specific T1w file found for {self.use_which_t1w} or more than one found.")
            t1w_file = t1w_files[0]
        else:
            print("No specific T1w file selected. Using the first one.")
            t1w_file = t1w_files[0]
        print(f"[T1_REGISTER] Using T1w file: {t1w_file}")

        template_filename, register_between_stripped = self._get_template_filename()

        # Create the workflow
        t1_register_wf = Workflow(name='t1_register_workflow')

        inputnode = Node(IdentityInterface(fields=['t1', 'mni_template', 't1_mni_out', 't1_2_mni_warp', 'mni_2_t1_warp']),
                         name='inputnode')

        mni_template = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'standard', 'MNI152', template_filename)
        space = self.template_space
        inputnode.inputs.t1 = t1w_file
        inputnode.inputs.mni_template = mni_template
        inputnode.inputs.register_between_stripped = register_between_stripped
        inputnode.inputs.t1_mni_out = os.path.join(self.output_path, rename_bids_file(t1w_file, {'space': space}, 'T1w', '.nii.gz'))
        inputnode.inputs.t1_2_mni_warp = os.path.join(self.output_path, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-T1w_to-{space}_warp.nii.gz')
        inputnode.inputs.mni_2_t1_warp = os.path.join(self.output_path, f'sub-{self.subject.subject_id}_ses-{self.session.session_id}_from-{space}_to-T1w_warp.nii.gz')

        print(f"[T1_REGISTER] Using MNI template: {template_filename} (register_between_stripped={register_between_stripped})")

        register_node = Node(SynthmorphNonlinear(), name='synthmorph_register')
        t1_register_wf.connect(inputnode, 't1', register_node, 't1')
        t1_register_wf.connect(inputnode, 'mni_template', register_node, 'mni_template')
        t1_register_wf.connect(inputnode, 't1_mni_out', register_node, 't1_mni_out')
        t1_register_wf.connect(inputnode, 't1_2_mni_warp', register_node, 't1_2_mni_warp')
        t1_register_wf.connect(inputnode, 'mni_2_t1_warp', register_node, 'mni_2_t1_warp')
        t1_register_wf.connect(inputnode, 'register_between_stripped', register_node, 'register_between_stripped')
        register_node.inputs.t1_stripped_out = os.path.join(self.output_path, rename_bids_file(t1w_file, {'space': 'T1w', 'desc': 'brain'}, 'T1w', '.nii.gz'))
        register_node.inputs.brain_mask_out = os.path.join(self.output_path, rename_bids_file(t1w_file, {'space': 'T1w', 'desc': 'brain'}, 'mask', '.nii.gz'))

        return t1_register_wf
