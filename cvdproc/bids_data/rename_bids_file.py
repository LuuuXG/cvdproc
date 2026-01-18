import os
import re

def rename_bids_file(original_filename, entities, suffix, extension):
    """
    Rename file to comply with BIDS requirements, automatically extract all entities, and allow overriding existing entities and suffix.
    :param original_filename: str, original filename (e.g., sub-AF1000077_ses-01_T1w.nii.gz)
    :param entities: dict, containing BIDS entities (e.g., {'subject': 'AF1000077', 'session': '01'})
    :param suffix: suffix (e.g., 'PVSlabel')
    :param extension: file extension (e.g., 'nii.gz')
    :return: new filename compliant with BIDS requirements
    """
    # Parse the original filename
    basename = os.path.basename(original_filename)  # Get the filename without the path
    name, ext = os.path.splitext(basename)  # Split the filename and extension

    # If the file extension contains .gz, continue splitting
    if ext == '.gz':
        name, ext2 = os.path.splitext(name)
        ext = ext2 + ext  # Combine .gz with the previous extension to form the complete extension

    # Define regular expressions to extract entity parts from the filename
    entity_patterns = {
        'sub': r"sub-([^_]+)",  # Match content after sub until the first underscore
        'ses': r"ses-([^_]+)",  # Match content after ses until the first underscore
        'sample': r"sample-([^_]+)",
        'task': r"task-([^_]+)",
        'tracksys': r"tracksys-([^_]+)",
        'acq': r"acq-([^_]+)",
        'nuc': r"nuc-([^_]+)",
        'voi': r"voi-([^_]+)",
        'ce': r"(?<!spa)ce-([^_]+)",
        'trc': r"trc-([^_]+)",
        'stain': r"stain-([^_]+)",
        'rec': r"rec-([^_]+)",
        'dir': r"dir-([^_]+)",
        'run': r"run-([^_]+)",
        'mod': r"mod-([^_]+)",
        'echo': r"echo-([^_]+)",
        'flip': r"flip-([^_]+)",
        'inv': r"inv-([^_]+)",
        'mt': r"mt-([^_]+)",
        'part': r"part-([^_]+)",
        'proc': r"proc-([^_]+)",
        'hemi': r"hemi-([^_]+)",
        'space': r"space-([^_]+)",
        'split': r"split-([^_]+)",
        'recording': r"recording-([^_]+)",
        'chunk': r"chunk-([^_]+)",
        'seg': r"seg-([^_]+)",
        'res': r"res-([^_]+)",
        'den': r"den-([^_]+)",
        'label': r"label-([^_]+)",
        'from': r"from-([^_]+)",
        'to': r"to-([^_]+)",
        'model': r"model-([^_]+)",
        'param': r"param-([^_]+)",
        'bundle': r"bundle-([^_]+)",
        'desc': r"desc-([^_]+)"
    }

    # Create a dictionary to store entities extracted from the filename
    extracted_entities = {}

    # Use regular expressions to extract each entity from the filename
    for entity, pattern in entity_patterns.items():
        match = re.search(pattern, name)
        if match:
            extracted_entities[entity] = match.group(1)

    # Use new entities to override original entities, if not provided, keep the original values
    subject_id = entities.get('sub', extracted_entities.get('sub', None))
    session_id = entities.get('ses', extracted_entities.get('ses', None))
    sample_id = entities.get('sample', extracted_entities.get('sample', None))
    task_id = entities.get('task', extracted_entities.get('task', None))
    tracksys_id = entities.get('tracksys', extracted_entities.get('tracksys', None))
    acq_id = entities.get('acq', extracted_entities.get('acq', None))
    nuc_id = entities.get('nuc', extracted_entities.get('nuc', None))
    voi_id = entities.get('voi', extracted_entities.get('voi', None))
    ce_id = entities.get('ce', extracted_entities.get('ce', None))
    trc_id = entities.get('trc', extracted_entities.get('trc', None))
    stain_id = entities.get('stain', extracted_entities.get('stain', None))
    rec_id = entities.get('rec', extracted_entities.get('rec', None))
    dir_id = entities.get('dir', extracted_entities.get('dir', None))
    run_id = entities.get('run', extracted_entities.get('run', None))
    mod_id = entities.get('mod', extracted_entities.get('mod', None))
    echo_id = entities.get('echo', extracted_entities.get('echo', None))
    flip_id = entities.get('flip', extracted_entities.get('flip', None))
    inv_id = entities.get('inv', extracted_entities.get('inv', None))
    mt_id = entities.get('mt', extracted_entities.get('mt', None))
    part_id = entities.get('part', extracted_entities.get('part', None))
    proc_id = entities.get('proc', extracted_entities.get('proc', None))
    hemi_id = entities.get('hemi', extracted_entities.get('hemi', None))
    space_id = entities.get('space', extracted_entities.get('space', None))
    split_id = entities.get('split', extracted_entities.get('split', None))
    recording_id = entities.get('recording', extracted_entities.get('recording', None))
    chunk_id = entities.get('chunk', extracted_entities.get('chunk', None))
    seg_id = entities.get('seg', extracted_entities.get('seg', None))
    res_id = entities.get('res', extracted_entities.get('res', None))
    den_id = entities.get('den', extracted_entities.get('den', None))
    label_id = entities.get('label', extracted_entities.get('label', None))
    from_id = entities.get('from', extracted_entities.get('from', None))
    to_id = entities.get('to', extracted_entities.get('to', None))
    model_id = entities.get('model', extracted_entities.get('model', None))
    param_id = entities.get('param', extracted_entities.get('param', None))
    bundle_id = entities.get('bundle', extracted_entities.get('bundle', None))
    desc_id = entities.get('desc', extracted_entities.get('desc', None))

    # Generate new filename
    new_filename_parts = []

    # Add optional entity parts
    if subject_id:
        new_filename_parts.append(f"sub-{subject_id}")
    if session_id:
        new_filename_parts.append(f"ses-{session_id}")
    if sample_id:
        new_filename_parts.append(f"sample-{sample_id}")
    if task_id:
        new_filename_parts.append(f"task-{task_id}")
    if tracksys_id:
        new_filename_parts.append(f"tracksys-{tracksys_id}")
    if acq_id:
        new_filename_parts.append(f"acq-{acq_id}")
    if nuc_id:
        new_filename_parts.append(f"nuc-{nuc_id}")
    if voi_id:
        new_filename_parts.append(f"voi-{voi_id}")
    if ce_id:
        new_filename_parts.append(f"ce-{ce_id}")
    if trc_id:
        new_filename_parts.append(f"trc-{trc_id}")
    if stain_id:
        new_filename_parts.append(f"stain-{stain_id}")
    if rec_id:
        new_filename_parts.append(f"rec-{rec_id}")
    if dir_id:
        new_filename_parts.append(f"dir-{dir_id}")
    if run_id:
        new_filename_parts.append(f"run-{run_id}")
    if mod_id:
        new_filename_parts.append(f"mod-{mod_id}")
    if echo_id:
        new_filename_parts.append(f"echo-{echo_id}")
    if flip_id:
        new_filename_parts.append(f"flip-{flip_id}")
    if inv_id:
        new_filename_parts.append(f"inv-{inv_id}")
    if mt_id:
        new_filename_parts.append(f"mt-{mt_id}")
    if part_id:
        new_filename_parts.append(f"part-{part_id}")
    if proc_id:
        new_filename_parts.append(f"proc-{proc_id}")
    if hemi_id:
        new_filename_parts.append(f"hemi-{hemi_id}")
    if space_id:
        new_filename_parts.append(f"space-{space_id}")
    if split_id:
        new_filename_parts.append(f"split-{split_id}")
    if recording_id:
        new_filename_parts.append(f"recording-{recording_id}")
    if chunk_id:
        new_filename_parts.append(f"chunk-{chunk_id}")
    if seg_id:
        new_filename_parts.append(f"seg-{seg_id}")
    if res_id:
        new_filename_parts.append(f"res-{res_id}")
    if den_id:
        new_filename_parts.append(f"den-{den_id}")
    if label_id:
        new_filename_parts.append(f"label-{label_id}")
    if from_id:
        new_filename_parts.append(f"from-{from_id}")
    if to_id:
        new_filename_parts.append(f"to-{to_id}")
    if model_id:
        new_filename_parts.append(f"model-{model_id}")
    if param_id:
        new_filename_parts.append(f"param-{param_id}")
    if bundle_id:
        new_filename_parts.append(f"bundle-{bundle_id}")
    if desc_id:
        new_filename_parts.append(f"desc-{desc_id}")

    new_filename_parts.append(f"{suffix}")

    # Join the new filename parts
    new_filename = "_".join(new_filename_parts) + f"{extension}"

    return new_filename
