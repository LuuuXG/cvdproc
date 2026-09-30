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
from .controllers.pipeline_manager import PipelineManager
from .controllers.pipeline_runner import run_pipeline_batch
from .guides import show_agent_next_steps

from cvdproc.bids_data.subject import BIDSSubject
from cvdproc.bids_data.session import BIDSSession

# as a function
def pipeline_input(bids_dir, subject_id, session_id, output_path):
    from cvdproc.bids_data.subject import BIDSSubject
    from cvdproc.bids_data.session import BIDSSession

    subject = BIDSSubject(subject_id, bids_dir)

    session = next((s for s in subject.get_all_sessions() if s.session_id == session_id), None) if session_id else None

    return subject, session, output_path

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
    parser.add_argument("--n_jobs", type=int, default=1, help="Maximum concurrent subject/session pipeline runs (default: 1)")
    parser.add_argument("--extract_results", action="store_true", help="Extract analysis results for all subjects.")
    parser.add_argument("--pipeline", type=str, help="Pipeline to run (e.g., 'wmh_quantification')")

    # Deprecated parameters
    #parser.add_argument("--nproc", type=int, help="Number of processors (not recommended)")
    parser.add_argument("--output_path", type=str, help="Override default output path (not recommended)")

    args = parser.parse_args()

    if args.n_jobs < 1:
        parser.error("--n_jobs must be a positive integer.")
    if args.run_pipeline:
        for name in ("config_file", "pipeline", "subject_id", "session_id"):
            if not getattr(args, name):
                parser.error(f"--run_pipeline requires --{name}.")
        if len(args.subject_id) != len(args.session_id):
            parser.error("--subject_id and --session_id must have the same number of values; no values are inferred.")

    if args.run_dcm2bids:
        for name in ("config_file", "subject_id", "session_id"):
            if not getattr(args, name):
                parser.error(f"--run_dcm2bids requires --{name}.")
        if bool(args.dicom_dir) == bool(args.dicom_subdir):
            parser.error("Provide exactly one of --dicom_dir or --dicom_subdir.")
        source_option = "dicom_dir" if args.dicom_dir else "dicom_subdir"
        sources = getattr(args, source_option)
        if not (len(args.subject_id) == len(args.session_id) == len(sources)):
            parser.error(f"--subject_id, --session_id and --{source_option} must have the same number of values "
                         f"(got {len(args.subject_id)}, {len(args.session_id)}, {len(sources)}). "
                         "Provide one session ID per subject; values are not repeated automatically.")

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
            "sub-*/ses-*/dwimap/",
            "sub-*/ses-*/**/*desc-XXX*"
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
            dicom_dirs = args.dicom_dir
        else:
            dicom_dirs = [os.path.join(bids_dir, "sourcedata", sub) for sub in args.dicom_subdir]
        
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

            # Optional: resample to iso-resolution
            resample_iso = dcm2bids_config.get("resample_to_iso", False)
            if resample_iso:
                processor.resample_to_iso(subject_id, session_id, resample_iso)

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
        pipeline_config = config.get("pipelines", {}).get(args.pipeline, {})
        if not pipeline_config:
            raise ValueError(f"No configuration found for pipeline '{args.pipeline}' in the configuration file.")
        visits = list(zip(args.subject_id, args.session_id))
        print_boxed_message_rich(f"Running '{args.pipeline}' for {len(visits)} visits (up to {min(args.n_jobs, len(visits))} at once)...", color="bold cyan")
        run_pipeline_batch(args.pipeline, visits, config["bids_dir"], config.get("output_dir", "./output"),
                           pipeline_config, matlab_path=config.get("matlab_path"), n_jobs=args.n_jobs)
        print_boxed_message_rich(f"All {len(visits)} visits finished!", color="bold cyan")

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

    if args.run_initialization:
        show_agent_next_steps(args.bids_dir)


def print_boxed_message_rich(message, color="cyan"):
    console = Console()
    console.print(Panel(message, style=color, expand=False))

if __name__ == "__main__":
    main_entry()
