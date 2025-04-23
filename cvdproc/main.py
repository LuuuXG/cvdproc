import argparse
import os
import yaml
from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import DataSink
from .pipelines.dcm2bids.dcm2bids_processor import Dcm2BidsProcessor
from .bids_data.subject import BIDSSubject
from .pipelines.common.input_for_pipeline import pipeline_input
from .controllers.pipeline_manager import PipelineManager
from .controllers.get_pipeline_workflow import get_pipeline_workflow
from .controllers.run_pipeline_workflow import run_pipeline_workflow

def load_config(config_file):
    """加载 YAML 配置文件"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main_entry():
    """
    Entry point for the command line script 'cvdproc'.
    """
    main()

def main():
    parser = argparse.ArgumentParser(description="Pipeline for DICOM to BIDS conversion and BIDS analysis")

    # 通用参数
    parser.add_argument("--config_file", type=str, required=False, help="Path to the main configuration file")
    parser.add_argument("--bids_dir", type=str, required=False, help="Path to the BIDS root directory")
    parser.add_argument("--subject_id", type=str, nargs='+', required=False, help="Subject IDs (e.g., '01 02')")
    parser.add_argument("--session_id", type=str, nargs='+', required=False, help="Session IDs (e.g., '01 02')")
    parser.add_argument("--run_initialization", action="store_true", help="Run BIDS initialization")
    parser.add_argument("--run_dcm2bids", action="store_true", help="Run DICOM to BIDS conversion")
    parser.add_argument("--dicom_subdir", type=str, help="Relative path to the DICOM folder under sourcedata (e.g., 'sub-1-dicom')")
    parser.add_argument("--check_data", action="store_true", help="Check the presence of specific data in the BIDS directory.")
    parser.add_argument("--run_pipeline", action="store_true", help="Run a BIDS-based analysis pipeline")
    parser.add_argument("--extract_results", action="store_true", help="Extract analysis results for all subjects.")
    parser.add_argument("--pipeline", type=str, help="Pipeline to run (e.g., 'wmh_quantification')")
    parser.add_argument("--nproc", type=int, help="Number of processors")
    parser.add_argument("--output_path", type=str, help="Override default output path")

    args = parser.parse_args()

    # === BIDS 初始化 ===
    if args.run_initialization:
        print("Initializing BIDS directory...")
        bids_dir = args.bids_dir
        processor = Dcm2BidsProcessor(bids_dir)
        processor.initialize()
        print("BIDS initialization completed.")

    # === DICOM 转 BIDS ===
    if args.run_dcm2bids:
        # 加载顶层配置文件
        config = load_config(args.config_file)

        if not all([args.subject_id, args.session_id, args.dicom_subdir]):
            parser.error("To run dcm2bids, please provide --subject_id, --session_id, and --dicom_subdir.")

        # 由于输入是list，转换为args.subject_id和args.session_id为str
        # 遍历每个subject_id和session_id，分别进行DICOM转BIDS
        for i in range(len(args.subject_id)):
            subject_id = args.subject_id[i]
            session_id = args.session_id[i] if args.session_id else ""

            bids_dir = config["bids_dir"]
            dicom_dir = os.path.join(bids_dir, "sourcedata", args.dicom_subdir)
            dcm2bids_config = config.get("dcm2bids", {})
            if not os.path.exists(dicom_dir):
                raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

            print(f"Running DICOM to BIDS conversion for subject {args.subject_id}, session {args.session_id}...")
            processor = Dcm2BidsProcessor(config["bids_dir"])
            # processor.initialize()
            processor.convert(
                config_file=dcm2bids_config["config_file"],
                dicom_directory=dicom_dir,
                subject_id=subject_id,
                session_id=session_id
            )
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
        # Load config
        config = load_config(args.config_file)
        global_matlab_path = config.get("matlab_path", None)
        global_spm_path = config.get("spm_path", None)

        # Check if pipeline is specified
        if not args.pipeline:
            parser.error("To run a pipeline, please provide --pipeline.")

        # Check if subject and session IDs match (if session IDs are provided)
        if len(args.session_id) > 0 and len(args.subject_id) != len(args.session_id):
            parser.error("Number of subject IDs and session IDs must match.")
        
        # Get the pipeline config
        pipeline_config = config.get("pipelines", {}).get(args.pipeline, {})
        if not pipeline_config:
            raise ValueError(f"No configuration found for pipeline '{args.pipeline}' in the configuration file.")
        
        # Create default output path
        subject_ids = args.subject_id
        session_ids = args.session_id
        default_output_paths = [
            os.path.join(
                config.get("output_dir", "./output"),
                args.pipeline,
                f"sub-{subject_ids[i]}",
                f"ses-{session_ids[i]}" if session_ids else ""
            )
            for i in range(len(subject_ids))
        ]

        ##############################
        # CVDproc Pipelines Workflow #
        ##############################
        cvdproc_wf = Workflow(name=f"cvdproc_{args.pipeline}")

        ###################################################
        # infosource: information source for the workflow #
        ###################################################
        infosource = Node(IdentityInterface(fields=["subject_id", "session_id", "output_path"]), name="infosource")
        # Use nipype iterables to iterate over subject and session IDs
        infosource.iterables = [
            ("subject_id", args.subject_id), 
            ("session_id", args.session_id),
            ("output_path", default_output_paths)
            ]
        
        infosource.synchronize = True
        
        ###############################################
        # input4pipeline: input node for the pipeline #
        ###############################################
        input4pipeline = Node(Function(
                input_names=["bids_dir", "subject_id", "session_id", "output_path"],
                output_names=["subject", "session", "output_path"],
                function=pipeline_input
        ), name="input4pipeline")

        # Connect infosource outputs to input4pipeline inputs
        input4pipeline.inputs.bids_dir = config["bids_dir"]
        cvdproc_wf.connect(infosource, "subject_id", input4pipeline, "subject_id")
        cvdproc_wf.connect(infosource, "session_id", input4pipeline, "session_id")
        cvdproc_wf.connect(infosource, "output_path", input4pipeline, "output_path")

        ##############################################
        # pipeline_workflow: workflow for a pipeline #
        ##############################################
        pipeline_workflow = Node(Function(
            input_names=["pipeline", "subject", "session", "output_path", "matlab_path", "spm_path", "pipeline_config"],
            output_names=["pipeline_workflow"],
            function=get_pipeline_workflow
        ), name='pipeline_workflow')

        # Connect input4pipeline outputs to pipeline_workflow inputs
        pipeline_workflow.inputs.pipeline = args.pipeline
        cvdproc_wf.connect(input4pipeline, "subject", pipeline_workflow, "subject")
        cvdproc_wf.connect(input4pipeline, "session", pipeline_workflow, "session")
        cvdproc_wf.connect(input4pipeline, "output_path", pipeline_workflow, "output_path")
        pipeline_workflow.inputs.matlab_path = global_matlab_path
        pipeline_workflow.inputs.spm_path = global_spm_path
        pipeline_workflow.inputs.pipeline_config = pipeline_config

        #############################
        # run the pipeline workflow #
        #############################
        run_workflow = Node(Function(
            input_names=["pipeline_workflow"],
            function=run_pipeline_workflow
        ), name="run_workflow")

        # Connect pipeline_workflow outputs to run_workflow inputs
        cvdproc_wf.connect(pipeline_workflow, "pipeline_workflow", run_workflow, "pipeline_workflow")

        if args.nproc:
            cvdproc_wf.run('MultiProc', plugin_args={'n_procs': args.nproc})
        else:
            cvdproc_wf.run()

    # === 提取结果 ===
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


if __name__ == "__main__":
    main_entry()