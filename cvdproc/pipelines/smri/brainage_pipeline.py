import os
import pandas as pd
import subprocess
from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface, Merge, Function

from cvdproc.pipelines.smri.brainage.brainage_nipype import BrainAgeR

class BrainAgePipeline:
    def __init__(self,
                 subject: object,
                 session: object,
                 output_path: str,
                 use_which_t1w: str = "T1w",
                 method: str = "brainageR",
                 extract_from: str = None,
                 **kwargs):
        """
        Brain Age Pipeline

        Args:
            subject (object): Subject object
            session (object): Session object
            output_path (str): Output path
            use_which_t1w (str, optional): Which T1w file to use. Defaults to 'T1w'
            method (str, optional): Method to use for brain age estimation. Defaults to 'brainageR'
            extract_from (str, optional): Path to extract results from. Defaults to None
        """
        self.subject = subject
        self.session = session
        self.output_path = output_path
        self.use_which_t1w = use_which_t1w
        self.method = method
        self.extract_from = extract_from
        self.kwargs = kwargs
    
    def check_data_requirements(self):
        return self.session.get_t1w_files() is not None

    def create_workflow(self):
        # Get the T1w file
        t1w_files = self.session.get_t1w_files()

        if self.use_which_t1w:
            t1w_files = [f for f in t1w_files if self.use_which_t1w in f]
            # ensure that there is only 1 suitable file
            if len(t1w_files) != 1:
                raise FileNotFoundError(f"No specific T1w file found for {self.use_which_t1w} or more than one found.")
            t1w_file = t1w_files[0]
        else:
            print("No specific T1w file selected. Using the first one.")
            t1w_files = [t1w_files[0]]
            t1w_file = t1w_files[0]
        print(f"[BRAIN AGE] Using T1w file: {t1w_file}")
        print(f"[BRAIN AGE] Using method: {self.method}")

        brainage_wf = Workflow(name=f"brainage_workflow")

        inputnode = Node(IdentityInterface(fields=['t1w_image']), name='inputnode')
        inputnode.inputs.t1w_image = t1w_file

        if self.method == "brainageR":
            brainageR_node = Node(BrainAgeR(), name='brainageR_node')
            brainage_wf.connect(inputnode, 't1w_image', brainageR_node, 't1w_image')
            brainageR_node.inputs.output_dir = os.path.join(self.output_path, "brainageR")
            brainageR_node.inputs.output_csv_filename = f"sub-{self.subject.subject_id}_ses-{self.session.session_id}_desc-brainageR_brainage.csv"
        else:
            raise ValueError(f"Unsupported brain age method: {self.method}")

        return brainage_wf

    def extract_results(self):
        os.makedirs(self.output_path, exist_ok=True)

        brainage_output_path = self.extract_from

        columns = ['Subject', 'Session', 'BrainAge_brainageR', 'BrainAge_LCI_brainageR', 'BrainAge_UCI_brainageR']
        results_df = pd.DataFrame(columns=columns)

        for subject_folder in os.listdir(brainage_output_path):
            subject_path = os.path.join(brainage_output_path, subject_folder)
            if not os.path.isdir(subject_path):
                continue

            session_folders = os.listdir(subject_path)
            for session_folder in session_folders:
                session_path = os.path.join(subject_path, session_folder)
                if not os.path.isdir(session_path):
                    continue

                brainageR_csv_path = os.path.join(session_path, "brainageR", f"{subject_folder}_{session_folder}_desc-brainageR_brainage.csv")

                if os.path.exists(brainageR_csv_path):
                    brainageR_df = pd.read_csv(brainageR_csv_path)
                    if {'brain.predicted_age', 'lower.CI', 'upper.CI'}.issubset(brainageR_df.columns):
                        new_data = pd.DataFrame([{
                            'Subject': subject_folder,
                            'Session': session_folder,
                            'BrainAge_brainageR': brainageR_df['brain.predicted_age'].values[0],
                            'BrainAge_LCI_brainageR': brainageR_df['lower.CI'].values[0],
                            'BrainAge_UCI_brainageR': brainageR_df['upper.CI'].values[0]
                        }])
                        results_df = pd.concat([results_df, new_data], ignore_index=True)
        
        # save excel outputs
        brain_age_results_path = os.path.join(self.output_path, "brainage_results.xlsx")
        
        if not results_df.empty:
            results_df.to_excel(brain_age_results_path, index=False)
            print(f"Brain age results saved to {brain_age_results_path}")

        return results_df