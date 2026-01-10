import os

from nipype import Node, Workflow, MapNode, Function
from nipype.interfaces.utility import IdentityInterface, Merge, Select

class ASLPipeline:
    def __init__(self,
                 subject: object,
                 session: object,
                 output_path: str,
                 use_which_asl: str = None,
                 use_which_t1w: str = None,
                 preprocess_method: str = 'ExploreASL'):
        
        self.subject = subject
        self.session = session
        self.output_path = output_path
        self.use_which_asl = use_which_asl
        self.use_which_t1w = use_which_t1w
        self.preprocess_method = preprocess_method

    def check_data_requirements(self):
        return self.session.get_perf_files() is not None and self.session.get_t1w_files() is not None
    
    def create_workflow(self):
        os.makedirs(self.output_path, exist_ok=True)

        # Get ASL and T1w files
        asl_files = self.session.get_perf_files()
        if self.use_which_asl is not None:
            nifti_asl_files = [f for f in asl_files if f.endswith('.nii') or f.endswith('.nii.gz')]
            asl_files = [f for f in nifti_asl_files if self.use_which_asl in f]
            if len(asl_files) != 1:
                raise ValueError(f"No specific ASL file found for {self.use_which_asl} or more than one found..")
            asl_file = asl_files[0]
            print(f"[ASL Pipeline] Using ASL file: {asl_file}")
        else:
            asl_file = asl_files[0]
            print(f"[ASL Pipeline] Using the first available ASL file: {asl_file}")
        
        t1w_files = self.session.get_t1w_files()
        if self.use_which_t1w is not None:
            t1w_files = [f for f in t1w_files if self.use_which_t1w in f]
            if len(t1w_files) != 1:
                raise ValueError(f"No specific T1w file found for {self.use_which_t1w} or more than one found..")
            t1w_file = t1w_files[0]
            print(f"[ASL Pipeline] Using T1w file: {t1w_file}") 
        else:
            t1w_file = t1w_files[0]
            print(f"[ASL Pipeline] Using the first available T1w file: {t1w_file}")
        
        # Handle M0 file (might have several volumes/4D):
        # 1. Separate M0 file: replace '_asl.nii' with '_m0scan.nii'
        # 2. Included in ASL file: read _aslcontext.tsv to find M0 index
        # 3. No M0 file
        m0_file = None
        m0_index = None
        asl_context_file = asl_file.replace('_asl.nii.gz', '_aslcontext.tsv').replace('_asl.nii', '_aslcontext.tsv')
        # the file should contain one signle column. Either have a 'volume_type' header or not.
        # possible volume types: 'control', 'label', 'm0scan', 'deltam', 'cbf', 'noRF', 'n/a'
        if os.path.exists(asl_context_file):
            with open(asl_context_file, 'r') as f:
                lines = f.readlines()
                if lines[0].strip() == 'volume_type':
                    volume_types = [line.strip() for line in lines[1:]]
                else:
                    volume_types = [line.strip() for line in lines]
            if 'm0scan' in volume_types:
                m0_index = volume_types.index('m0scan')
                print(f"[ASL Pipeline] Found M0 scan in ASL file at index {m0_index}.")
            else:
                # see if separate M0 file exists
                possible_m0_file = asl_file.replace('_asl.nii.gz', '_m0scan.nii.gz').replace('_asl.nii', '_m0scan.nii')
                if os.path.exists(possible_m0_file):
                    m0_file = possible_m0_file
                    print(f"[ASL Pipeline] Using separate M0 file: {m0_file}")

        # ===============================
        # Main Workflow
        # ===============================
        asl_wf = Workflow(name='asl_workflow')

        if self.preprocess_method.lower() == 'exploreasl':
            from cvdproc.pipelines.perfusion.exploreasl import ExploreASLNode
            asl_node = ExploreASLNode(asl_file=asl_file,
                                      t1w_file=t1w_file,
                                      m0_file=m0_file,
                                      m0_index=m0_index,
                                      output_path=self.output_path)
            

        return asl_wf