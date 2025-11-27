import os
import pandas as pd
import subprocess
import xml.etree.ElementTree as ET

class CAT12Pipeline:
    def __init__(self, subject, session, output_path, matlab_path=None, **kwargs):
        """
        CAT12 pipeline
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)

        self.use_which_t1w = kwargs.get('use_which_t1w', None)
        self.matlab_path = matlab_path or kwargs.get('matlab_path', 'matlab')
        self.job_name = kwargs.get('job', 'segmentation')
        self.cat12_path = kwargs.get('cat12_path', None)
        self.cat12_standalone_path = kwargs.get('cat12_standalone_path', None)
        self.extract_from = kwargs.get('extract_from', None)

    def check_data_requirements(self):
        """
        Check whether required data is available.
        :return: bool
        """
        return self.session.get_t1w_files() is not None

    def run(self):
        t1w_files = self.session.get_t1w_files()

        if self.use_which_t1w:
            t1w_files = [f for f in t1w_files if self.use_which_t1w in f]
            # Ensure exactly one suitable file remains
            if len(t1w_files) != 1:
                raise FileNotFoundError(f"No specific T1w file found for {self.use_which_t1w} or more than one found.")
            t1w_file = t1w_files[0]
        else:
            print("No specific T1w file selected. Using the first one.")
            t1w_files = [t1w_files[0]]
            t1w_file = t1w_files[0]

        if t1w_file is None:
            raise FileNotFoundError("No T1w file found in anat directory.")
        
        # create output directory
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        # copy t1w file to output directory
        t1w_file = os.path.abspath(t1w_file)
        t1w_file_new = os.path.join(self.output_path, os.path.basename(t1w_file))
        if not os.path.exists(t1w_file_new):
            subprocess.run(["cp", t1w_file, t1w_file_new])

        # CAT12 segmentation
        if self.job_name.lower() == 'segmentation':
            standalone_script_path = os.path.join(self.cat12_standalone_path, 'standalone', 'cat_standalone.sh')
            matlab_runtime = os.path.join(self.cat12_standalone_path, 'v93')
            seg_matlab_script_path = os.path.join(self.cat12_standalone_path, 'standalone', 'cat_standalone_segment_enigma.m')

            bash_command = [
                standalone_script_path,
                "-m", matlab_runtime,
                "-b", seg_matlab_script_path,
                "-a",
                (
                    "matlabbatch{1}.spm.tools.cat.estwrite.output.surface = 0; "
                    "matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.Schaefer2018_100Parcels_17Networks_order = 1; "
                    "matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.Schaefer2018_200Parcels_17Networks_order = 1; "
                    "matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.Schaefer2018_400Parcels_17Networks_order = 1; "
                    "matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.Schaefer2018_600Parcels_17Networks_order = 1; "
                    "matlabbatch{1}.spm.tools.cat.estwrite.extopts.registration.vox = 1;"
                ),
                t1w_file_new
            ]

            try:
                subprocess.run(bash_command, check=True)

                cat12_dafault_output_path = os.path.dirname(os.path.dirname(self.output_path))
                cat12_dafault_output_path = os.path.join(cat12_dafault_output_path, 'CAT12.9')

                cat12_log_path = os.path.dirname(os.path.dirname(os.path.dirname(cat12_dafault_output_path)))
                cat12_log_path = os.path.join(cat12_log_path, 'CAT12.9')
                #print("cat12_log_path: ", cat12_log_path)

                if os.path.exists(os.path.join(self.output_path, 'report')):
                    subprocess.run(["rm", "-r", os.path.join(self.output_path, 'report')])

                # move 'label', 'mri', 'report' folders in cat12_dafault_output_path to output path
                for folder in ['label', 'mri', 'report']:
                    folder_path = os.path.join(cat12_dafault_output_path, folder)
                    if os.path.exists(folder_path):
                        subprocess.run(["mv", folder_path, self.output_path])
                subprocess.run(["rm", "-r", cat12_dafault_output_path])

                # gzip all .nii in os.path.join(self.output_path, 'mri')
                for root, dirs, files in os.walk(os.path.join(self.output_path, 'mri')):
                    for file in files:
                        if file.endswith('.nii'):
                            subprocess.run(["gzip", os.path.join(root, file)])

                # delete cat12_log_path
                if os.path.exists(cat12_log_path):
                    subprocess.run(["rm", "-r", cat12_log_path])

                print("Preprocessing completed.")
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while running preprocessing script: {e}")

            # delete t1w file in output path
            if os.path.exists(t1w_file_new):
                subprocess.run(["rm", t1w_file_new])
            
            # TIV extraction
            # get file name (remove .nii.gz or .nii)
            
            t1w_basename = os.path.basename(t1w_file_new)
            if t1w_basename.endswith('.nii.gz'):
                t1w_filename = t1w_basename[:-7]
            elif t1w_file_new.endswith('.nii'):
                t1w_filename = t1w_basename[:-4]
            xml_path = os.path.join(self.output_path, 'report', f'cat_{t1w_filename}.xml')
            tiv_output_path = os.path.join(self.output_path, 'TIV.txt')

            tree = ET.parse(xml_path)
            root = tree.getroot()

            tiv = root.find(".//vol_TIV")
            csf = root.find(".//vol_abs_CGW")

            tiv_value = float(tiv.text) if tiv is not None else 0.0
            csf_values = [float(x) for x in csf.text.strip("[]").split()] if csf is not None else [0.0, 0.0, 0.0, 0.0]

            output_values = [tiv_value] + csf_values[:3]

            with open(tiv_output_path, "w") as file:
                for value in output_values:
                    file.write(f"{value} ")
    
    def _extract_tiv(self, subject_id, session_id, base_path):
        """
        Extract TIV, CSF, GM, and WM values.
        """
        txt_path = os.path.join(base_path, 'TIV.txt')

        # Initialize values
        tiv = None
        csf = None
        gm = None
        wm = None

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                # Read the first line
                line = f.readline().strip()
                # Split by spaces to extract values
                values = line.split()

                # Verify there are enough values
                if len(values) >= 4:
                    tiv = float(values[0])  # First value is TIV
                    csf = float(values[1])  # Second value is CSF
                    gm = float(values[2])   # Third value is GM
                    wm = float(values[3])   # Fourth value is WM

        # Return the extracted results
        return pd.DataFrame([{
            'Subject': subject_id,
            'Session': session_id,
            'TIV': tiv,
            'CSF': csf,
            'GM': gm,
            'WM': wm
        }])
    
    def extract_results(self):
        os.makedirs(self.output_path, exist_ok=True)

        cat12_output_path = self.extract_from

        columns = ['Subject', 'Session', 'TIV', 'CSF', 'GM', 'WM']
        results_df = pd.DataFrame(columns=columns)

        # Iterate over all sub-* folders
        for subject_folder in os.listdir(cat12_output_path):
            subject_id = subject_folder.split('-')[1]
            subject_folder_path = os.path.join(cat12_output_path, subject_folder)

            if os.path.isdir(subject_folder_path):
                # Check for ses-* folders
                session_folders = [f for f in os.listdir(subject_folder_path) if 'ses-' in f]

                if session_folders:  # If ses-* folders exist
                    for session_folder in session_folders:
                        session_path = os.path.join(subject_folder_path, session_folder)
                        new_data = self._extract_tiv(subject_id, session_folder.split('-')[1], session_path)
                        results_df = pd.concat([results_df, new_data], ignore_index=True)
                else:  # If there are no ses-* folders
                    new_data = self._extract_tiv(subject_id, 'N/A', subject_folder_path)
                    results_df = pd.concat([results_df, new_data], ignore_index=True)

        # Save results to an Excel file
        output_excel_path = os.path.join(self.output_path, 'cat12_tiv_results.xlsx')
        results_df.to_excel(output_excel_path, header=True, index=False)
        print(f"Quantification results saved to {output_excel_path}")
