import argparse
import os
import yaml
import json
from rich import print
from rich.console import Console
from rich.panel import Panel
import logging

# Clean up root logger handlers (to prevent duplicated log lines)
root_logger = logging.getLogger()
if root_logger.hasHandlers():
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import DataSink
from .pipelines.dcm2bids.dcm2bids_processor import Dcm2BidsProcessor
from .bids_data.subject import BIDSSubject
from .pipelines.common.input_for_pipeline import pipeline_input
from .controllers.pipeline_manager import PipelineManager

def load_config(config_file):
    """Load a YAML configuration file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main_entry():
    """
    Entry point for the command line script 'cvdproc'.
    """
    main()

def main():
    parser = argparse.ArgumentParser(description="Pipeline for DICOM to BIDS conversion and BIDS analysis")

    # Global parameters for the script
    parser.add_argument("--config_file", type=str, required=False, help="Path to the main configuration file")
    parser.add_argument("--bids_dir", type=str, required=False, help="Path to the BIDS root directory")
    parser.add_argument("--subject_id", type=str, nargs='+', required=False, help="Subject IDs (e.g., '01 02')")
    parser.add_argument("--session_id", type=str, nargs='+', required=False, help="Session IDs (e.g., '01 02')")
    parser.add_argument("--run_initialization", action="store_true", help="Run BIDS initialization")
    parser.add_argument("--run_dcm2bids", action="store_true", help="Run DICOM to BIDS conversion")
    parser.add_argument("--dicom_subdir", type=str, nargs='+', help="Relative path to the DICOM folder under sourcedata (e.g., 'sub-1-dicom')")
    parser.add_argument("--dicom_dir", type=str, nargs='+', help="Full path(s) to DICOM directory (alternative to --dicom_subdir)")
    parser.add_argument("--check_data", action="store_true", help="Check the presence of specific data in the BIDS directory.")
    parser.add_argument("--run_pipeline", action="store_true", help="Run a BIDS-based analysis pipeline")
    parser.add_argument("--extract_results", action="store_true", help="Extract analysis results for all subjects.")
    parser.add_argument("--pipeline", type=str, help="Pipeline to run (e.g., 'wmh_quantification')")

    # Deprecated parameters
    #parser.add_argument("--nproc", type=int, help="Number of processors (not recommended)")
    parser.add_argument("--output_path", type=str, help="Override default output path (not recommended)")

    args = parser.parse_args()

    # === BIDS Initialization ===
    if args.run_initialization:
        print("Initializing BIDS directory...")
        bids_dir = args.bids_dir
        processor = Dcm2BidsProcessor(bids_dir)
        processor.initialize()

        # add .bidsignore with additional common ignores
        bidsignore_file = os.path.join(bids_dir, '.bidsignore')
        additional_ignores = [
            "tmp_dcm2bids/",
            "sub-*/ses-*/swi/",
            "sub-*/ses-*/qsm/",
            "sub-*/ses-*/pwi/",
            "sub-*/ses-*/dwimap/"
        ]
        if os.path.exists(bidsignore_file):
            with open(bidsignore_file, 'r') as f:
                existing_ignores = f.read().splitlines()
        else:
            existing_ignores = []
        with open(bidsignore_file, 'a') as f:
            for ignore in additional_ignores:
                if ignore not in existing_ignores:
                    f.write(ignore + '\n')
        print(".bidsignore file updated.")

    # === DICOM to BIDS ===
    if args.run_dcm2bids:
        # Load main configuration
        config = load_config(args.config_file)

        dcm2bids_config = config.get("dcm2bids", {})
        bids_dir = config["bids_dir"]

        # Determine source of DICOM directories
        if args.dicom_dir:
            if len(args.dicom_dir) != len(args.subject_id):
                parser.error("The number of --dicom_dir must match the number of --subject_id.")
            dicom_dirs = args.dicom_dir
        elif args.dicom_subdir:
            if len(args.dicom_subdir) != len(args.subject_id):
                parser.error("The number of --dicom_subdir must match the number of --subject_id.")
            dicom_dirs = [os.path.join(bids_dir, "sourcedata", sub) for sub in args.dicom_subdir]
        else:
            parser.error("Please provide either --dicom_dir or --dicom_subdir.")
        
        dicom_dirname = os.path.basename(dicom_dirs[0])
        subjects_info = os.path.join(bids_dir, 'participants.tsv')

        for subject_id, session_id, dicom_dir in zip(args.subject_id, args.session_id, dicom_dirs):
            if not os.path.exists(dicom_dir):
                raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

            print(f"Running DICOM to BIDS conversion for subject {subject_id}, session {session_id}...")

            processor = Dcm2BidsProcessor(bids_dir)
            processor.convert(
                config_file=dcm2bids_config["config_file"],
                dicom_directory=dicom_dir,
                subject_id=subject_id,
                session_id=session_id,
                ignore_patterns=dcm2bids_config.get("ignore", []),
                keep_temp=dcm2bids_config.get("keep_filtered_dicom", False),
            )

            # --- Update: Extract the first DICOM file and update participants.tsv ---
            ds = processor.find_first_dicom(dicom_dir)
            if ds:
                processor.update_participants_tsv(bids_dir, subject_id, session_id, ds)
            else:
                print(f"Warning: No DICOM file found in {dicom_dir}, skipping participants.tsv update.")

            # Optional: fix bvec/bval
            fix_config = dcm2bids_config.get("dwi_fix_bvecbval", [])
            if fix_config:
                processor.fix_dwi_bvec_bval(subject_id, session_id, fix_config)
            
            # Optional: fix aslcontext
            aslcontext_config = dcm2bids_config.get("perf_fix_aslcontext", [])
            if aslcontext_config:
                processor.fix_perf_aslcontext(subject_id, session_id, aslcontext_config)

            # Optional: deface anat images
            # if dcm2bids_config 'deface_anat' is set to True
            deface_anat = dcm2bids_config.get("deface_anat", False)
            if deface_anat:
                processor.deface_anat(subject_id, session_id, suffix_list=['T1w', 'T2w', 'FLAIR'])

            # Optional: fix IntendedFor (for nipreps)
            # if dcm2bids_config 'fix_intendedfor' is set to True
            fix_intendedfor = dcm2bids_config.get("fix_intendedfor", False)
            if fix_intendedfor:
                processor.fix_intendedfor_for_subject_session(subject_id, session_id)

        print("DICOM to BIDS conversion completed.") 
    
    # === BIDS check ===
    if args.check_data:
        config = load_config(args.config_file)
        processor = Dcm2BidsProcessor(config["bids_dir"])
        check_data_config = config.get("check_data", [])

        if not check_data_config:
            parser.error("No 'check_data' configuration found in the config file.")

        processor = Dcm2BidsProcessor(config["bids_dir"])
        processor.check_data(check_data_config)

    # === BIDS Pipelines ===
    if args.run_pipeline:
        print_boxed_message_rich(f"Checking whether the arguments are correct...", color="bold cyan")

        config = load_config(args.config_file)
        global_matlab_path = config.get("matlab_path", None)
        global_spm_path = config.get("spm_path", None)

        if not args.pipeline:
            parser.error("To run a pipeline, please provide --pipeline.")

        if len(args.session_id) > 0 and len(args.subject_id) != len(args.session_id):
            parser.error("Number of subject IDs and session IDs must match.")

        pipeline_config = config.get("pipelines", {}).get(args.pipeline, {})
        if not pipeline_config:
            raise ValueError(f"No configuration found for pipeline '{args.pipeline}' in the configuration file.")

        subject_ids = args.subject_id
        session_ids = args.session_id
        bids_dir = config["bids_dir"]
        output_base = config.get("output_dir", "./output")
        numbers = len(args.subject_id)

        print(f"[green]Arguements seem correct[/green]")

        if numbers == 1:
            print_boxed_message_rich(f"Running '{args.pipeline}' for {numbers} visit...", color="bold cyan")
        else:
            print_boxed_message_rich(f"Running '{args.pipeline}' for {numbers} visits...", color="bold cyan")
        for i in range(len(subject_ids)):
            sub_id = subject_ids[i]
            ses_id = session_ids[i] if session_ids else None

            if session_ids is not None:
                print(f"[green]Currently processing: sub-{subject_ids[i]} ses-{session_ids[i]}[/green]")
            else:
                print(f"[green]Currently processing: sub-{subject_ids[i]}[/green]")
            
            print(f"[green]Stage 1: Create nipype workflow for {args.pipeline}[/green]")

            # Create BIDSSubject and BIDSSession instances
            subject = BIDSSubject(sub_id, bids_dir)
            session = None
            if ses_id:
                session = next((s for s in subject.get_all_sessions() if s.session_id == ses_id), None)
                if session is None:
                    raise ValueError(f"No session {ses_id} found for subject {sub_id}")

            # Set output path
            output_path = os.path.join(output_base, args.pipeline, f"sub-{sub_id}", f"ses-{ses_id}" if ses_id else "")
            os.makedirs(output_path, exist_ok=True)

            # Get pipeline object and create workflow
            manager = PipelineManager()
            pipeline = manager.get_pipeline(
                args.pipeline,
                subject=subject,
                session=session,
                output_path=output_path,
                matlab_path=global_matlab_path,
                spm_path=global_spm_path,
                **pipeline_config
            )

            wf = pipeline.create_workflow()
            wf.base_dir = os.path.join(bids_dir, "derivatives", "workflows", f"sub-{sub_id}", f"ses-{ses_id}" if ses_id else "")

            print(f"[green]Stage 2: Running nipype workflow for {args.pipeline}[/green]")
            wf.run()

            print(f"[green]Finished processing sub-{sub_id} ses-{ses_id if ses_id else 'N/A'}[/green]")
        
        if numbers == 1:
            print_boxed_message_rich(f"Finished!", color="bold cyan")
        else:
            print_boxed_message_rich(f"All {numbers} visits finished!", color="bold cyan")

    # === Extract results ===
    if args.extract_results:
        config = load_config(args.config_file)
        
        if not args.pipeline:
            parser.error("To extract results, please provide --pipeline.")

        pipeline_config = config.get("pipelines", {}).get(args.pipeline, {})
        if not pipeline_config:
            raise ValueError(f"No configuration found for pipeline '{args.pipeline}' in the configuration file.")
        
        default_output_path = os.path.join(
            config.get("output_dir", "./output"),
            "population",
            args.pipeline
        )
        output_path = args.output_path or default_output_path

        manager = PipelineManager()
        pipeline = manager.get_pipeline(
            args.pipeline,
            subject=None,
            session=None,
            output_path=output_path, # where to save the extracted results
            **pipeline_config
        )

        pipeline.extract_results()
        print(f"Results extracted for pipeline '{args.pipeline}'. Results saved to {output_path}.")

    if not args.run_dcm2bids and not args.run_pipeline and not args.run_initialization and not args.extract_results and not args.check_data:
        print("No action specified. Use --run_dcm2bids, --run_pipeline, --extract_results or --run_initialization.")


def print_boxed_message_rich(message, color="cyan"):
    console = Console()
    console.print(Panel(message, style=color, expand=False))

if __name__ == "__main__":
    main_entry()