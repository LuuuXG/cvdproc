from __future__ import division

import os
import nipype.pipeline.engine as pe
from nipype import SelectFiles
import nipype.interfaces.utility as util
from nipype import IdentityInterface

from .utils import *

def create_bullseye_pipeline(
    scans_dir,
    work_dir,
    outputdir,
    subject_ids,
    name='bullseye_pipeline',
    lobes_fname='lobes_wmparc.nii.gz',
    shells_fname='shells_wmparc.nii.gz',
    bullseye_fname='bullseye_wmparc.nii.gz'
):
    """
    Build a workflow that generates three WM parcellations and writes them under:
        <outputdir>/bullseye/<custom_filenames>

    Parameters
    ----------
    scans_dir : str
        FreeSurfer SUBJECTS_DIR (each subject has mri/, surf/, label/ etc.)
    work_dir : str
        Nipype working directory.
    outputdir : str
        Base output directory. Results will be placed in "<outputdir>/bullseye".
    subject_ids : list[str] or str
        Subject id(s). If a single subject id string is passed, it is used as is.
    name : str
        Workflow name.
    lobes_fname : str
        File name for the lobar WM parcellation (default: 'lobes_wmparc.nii.gz').
    shells_fname : str
        File name for the depth shells parcellation (default: 'shells_wmparc.nii.gz').
    bullseye_fname : str
        File name for the final bullseye parcellation (default: 'bullseye_wmparc.nii.gz').

    Returns
    -------
    bullwf : nipype Workflow
        The configured workflow.
    Notes
    -----
    The workflow exposes an 'outputnode' with fields:
        - lobes_wmparc
        - shells_wmparc
        - bullseye_wmparc
    which contain absolute paths to the generated files.
    """

    # Ensure FreeSurfer SUBJECTS_DIR
    os.environ['SUBJECTS_DIR'] = scans_dir

    # Ensure output subfolder exists
    bullseye_dir = os.path.abspath(os.path.join(outputdir, 'bullseye'))
    os.makedirs(bullseye_dir, exist_ok=True)

    # Compose absolute output paths with custom file names
    out_lobes_path = os.path.join(bullseye_dir, lobes_fname)
    out_shells_path = os.path.join(bullseye_dir, shells_fname)
    out_bullseye_path = os.path.join(bullseye_dir, bullseye_fname)

    # Workflow
    bullwf = pe.Workflow(name=name)
    bullwf.base_dir = work_dir

    # I/O nodes
    inputnode = pe.Node(
        IdentityInterface(fields=['subject_ids',
                                  'out_lobes_path',
                                  'out_shells_path',
                                  'out_bullseye_path']),
        name='inputnode'
    )
    inputnode.inputs.subject_ids = subject_ids
    inputnode.inputs.out_lobes_path = out_lobes_path
    inputnode.inputs.out_shells_path = out_shells_path
    inputnode.inputs.out_bullseye_path = out_bullseye_path

    outputnode = pe.Node(
        IdentityInterface(fields=['lobes_wmparc', 'shells_wmparc', 'bullseye_wmparc']),
        name='outputnode'
    )

    # Select inputs
    template = {
        "ASEG": "{subject_id}/mri/aseg*gz",
        "RIBBON": "{subject_id}/mri/ribbon.mgz",
        "ANNOT_LH": "{subject_id}/label/lh.aparc.annot",
        "ANNOT_RH": "{subject_id}/label/rh.aparc.annot",
        "WHITE_LH": "{subject_id}/surf/lh.white",
        "WHITE_RH": "{subject_id}/surf/rh.white",
        "PIAL_LH": "{subject_id}/surf/lh.pial",
        "PIAL_RH": "{subject_id}/surf/rh.pial",
        "subject_id": "{subject_id}"
    }
    fileselector = pe.Node(SelectFiles(template), name='fileselect')
    fileselector.inputs.base_directory = scans_dir

    # Lobar parcellation on surfaces
    annot2label_lh = pe.Node(interface=Annot2Label(), name='annot2label_lh')
    annot2label_lh.inputs.hemi = 'lh'
    annot2label_lh.inputs.lobes = 'lobes'

    annot2label_rh = pe.Node(interface=Annot2Label(), name='annot2label_rh')
    annot2label_rh.inputs.hemi = 'rh'
    annot2label_rh.inputs.lobes = 'lobes'

    # Map lobes to volume (WM)
    aparc2aseg = pe.Node(interface=Aparc2Aseg(), name='aparc2aseg')
    aparc2aseg.inputs.annot = 'lobes'
    aparc2aseg.inputs.labelwm = True
    aparc2aseg.dmax = 1000
    aparc2aseg.inputs.rip = True
    aparc2aseg.inputs.hypo = True
    # Temporary file name here is fine; we will produce final outputs later
    aparc2aseg.inputs.out_file = 'lobes+aseg.nii.gz'

    # Group lobes and discard others
    filter_lobes = pe.Node(
        interface=util.Function(
            input_names=['in_file', 'include_superlist', 'fixed_id', 'map_pairs_list'],
            output_names=['out_file'],
            function=filter_labels
        ),
        name='filter_lobes'
    )
    # Include insula with frontal; exclude the long superior band
    filter_lobes.inputs.include_superlist = [
        [3001, 3007], [4001, 4007], [3004], [4004], [3005], [4005], [3006], [4006]
    ]
    filter_lobes.inputs.fixed_id = None
    filter_lobes.inputs.map_pairs_list = [
        [3001, 11], [4001, 21], [3004, 12], [4004, 22],
        [3005, 13], [4005, 23], [3006, 14], [4006, 24]
    ]

    # Masks
    ventricles_mask = pe.Node(
        interface=util.Function(
            input_names=['in_file', 'include_superlist', 'fixed_id', 'map_pairs_list'],
            output_names=['out_file'],
            function=filter_labels
        ),
        name='ventricles_mask'
    )
    ventricles_mask.inputs.include_superlist = [[43, 4]]
    ventricles_mask.inputs.fixed_id = [1]
    ventricles_mask.inputs.map_pairs_list = None

    cortex_mask = pe.Node(
        interface=util.Function(
            input_names=['in_file', 'include_superlist', 'fixed_id', 'map_pairs_list'],
            output_names=['out_file'],
            function=filter_labels
        ),
        name='cortex_mask'
    )
    cortex_mask.inputs.include_superlist = [[1001, 2001, 1004, 2004, 1005, 2005, 1006, 2006]]
    cortex_mask.inputs.fixed_id = [1]
    cortex_mask.inputs.map_pairs_list = None

    bgt_mask = pe.Node(
        interface=util.Function(
            input_names=['in_file', 'include_superlist', 'fixed_id', 'map_pairs_list'],
            output_names=['out_file'],
            function=filter_labels
        ),
        name='bgt_mask'
    )
    bgt_mask.inputs.include_superlist = [[10, 49, 11, 12, 50, 51, 26, 58, 13, 52]]
    bgt_mask.inputs.fixed_id = [5]
    bgt_mask.inputs.map_pairs_list = None

    # Normalized distance map (ventricles -> cortex)
    ndist_map = pe.Node(
        interface=util.Function(
            input_names=['orig_file', 'dest_file'],
            output_names=['out_file'],
            function=norm_dist_map
        ),
        name='ndist_map'
    )

    # Generate WM parcellation by filling discarded lobes and unsegmented WM
    gen_wmparc = pe.Node(
        interface=util.Function(
            input_names=['incl_file', 'ndist_file', 'label_file', 'incl_labels', 'verbose'],
            output_names=['out_file'],
            function=generate_wmparc
        ),
        name='gen_wmparc'
    )
    gen_wmparc.inputs.incl_labels = [3003, 4003, 5001, 5002]
    gen_wmparc.inputs.verbose = False

    # Merge lobes and BGT -> final lobar WM parcellation
    lobe_wmparc = pe.Node(
        interface=util.Function(
            input_names=['in1_file', 'in2_file', 'out_file', 'intersect'],
            output_names=['out_file'],
            function=merge_labels
        ),
        name='lobe_wmparc'
    )
    lobe_wmparc.inputs.intersect = False
    # Set absolute output path with user-defined file name
    lobe_wmparc.inputs.out_file = out_lobes_path

    # Depth shells
    depth_wmparc = pe.Node(
        interface=util.Function(
            input_names=['ndist_file', 'n_shells', 'out_file', 'mask_file'],
            output_names=['out_file'],
            function=create_shells
        ),
        name='depth_wmparc'
    )
    depth_wmparc.inputs.n_shells = 4
    depth_wmparc.inputs.out_file = out_shells_path

    # Final bullseye = intersect depth shells with lobar WM
    bullseye_wmparc = pe.Node(
        interface=util.Function(
            input_names=['in1_file', 'in2_file', 'out_file', 'intersect'],
            output_names=['out_file'],
            function=merge_labels
        ),
        name='bullseye_wmparc'
    )
    bullseye_wmparc.inputs.intersect = True
    bullseye_wmparc.inputs.out_file = out_bullseye_path

    # -------- Connections --------

    # Subject iteration / selection
    bullwf.connect(inputnode, 'subject_ids', fileselector, 'subject_id')

    # Surface lobes
    bullwf.connect(fileselector, 'subject_id', annot2label_lh, 'subject')
    bullwf.connect(fileselector, 'ANNOT_LH',   annot2label_lh, 'in_annot')
    bullwf.connect(fileselector, 'subject_id', annot2label_rh, 'subject')
    bullwf.connect(fileselector, 'ANNOT_RH',   annot2label_rh, 'in_annot')

    # Volume mapping
    bullwf.connect(annot2label_rh, 'out_annot_file', aparc2aseg, 'in_lobes_rh')
    bullwf.connect(annot2label_lh, 'out_annot_file', aparc2aseg, 'in_lobes_lh')
    bullwf.connect(fileselector,   'subject_id',     aparc2aseg, 'subject')

    # Filter & masks
    bullwf.connect(aparc2aseg, 'out_file', filter_lobes,    'in_file')
    bullwf.connect(aparc2aseg, 'out_file', ventricles_mask, 'in_file')
    bullwf.connect(aparc2aseg, 'out_file', cortex_mask,     'in_file')
    bullwf.connect(aparc2aseg, 'out_file', bgt_mask,        'in_file')

    # Distance map
    bullwf.connect(ventricles_mask, 'out_file', ndist_map, 'orig_file')
    bullwf.connect(cortex_mask,     'out_file', ndist_map, 'dest_file')

    # Generate WM parc
    bullwf.connect(aparc2aseg,   'out_file', gen_wmparc, 'incl_file')
    bullwf.connect(ndist_map,    'out_file', gen_wmparc, 'ndist_file')
    bullwf.connect(filter_lobes, 'out_file', gen_wmparc, 'label_file')

    # Lobar WM = gen_wmparc + BGT
    bullwf.connect(gen_wmparc, 'out_file', lobe_wmparc, 'in1_file')
    bullwf.connect(bgt_mask,   'out_file', lobe_wmparc, 'in2_file')

    # Depth shells over lobar WM mask
    bullwf.connect(ndist_map,   'out_file', depth_wmparc, 'ndist_file')
    bullwf.connect(lobe_wmparc, 'out_file', depth_wmparc, 'mask_file')

    # Final bullseye = intersection(lobar WM, depth shells)
    bullwf.connect(lobe_wmparc,  'out_file', bullseye_wmparc, 'in1_file')
    bullwf.connect(depth_wmparc, 'out_file', bullseye_wmparc, 'in2_file')

    # Expose final outputs via outputnode (absolute paths returned by the Function interfaces)
    bullwf.connect(lobe_wmparc,     'out_file', outputnode, 'lobes_wmparc')
    bullwf.connect(depth_wmparc,    'out_file', outputnode, 'shells_wmparc')
    bullwf.connect(bullseye_wmparc, 'out_file', outputnode, 'bullseye_wmparc')

    return bullwf
