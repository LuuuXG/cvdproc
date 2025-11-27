import os
import glob
from .session import BIDSSession
from .subject import BIDSSubject

class BIDSUtils:
    @staticmethod
    def list_sessions(bids_root, subject_label):
        """
        List all sessions for a given subject in a BIDS dataset.

        Parameters:
        bids_root (str): The root directory of the BIDS dataset.
        subject_label (str): The label of the subject (e.g., '01').

        Returns:
        list: A list of session labels (e.g., ['ses-01', 'ses-02']).
        """
        subject_path = os.path.join(bids_root, f"sub-{subject_label}")
        session_paths = glob.glob(os.path.join(subject_path, "ses-*"))
        sessions = [os.path.basename(ses_path) for ses_path in session_paths]
        return sessions

    @staticmethod
    def list_subjects(bids_root):
        """
        List all subjects in a BIDS dataset.

        Parameters:
        bids_root (str): The root directory of the BIDS dataset.

        Returns:
        list: A list of subject labels (e.g., ['01', '02']).
        """
        subject_paths = glob.glob(os.path.join(bids_root, "sub-*"))
        subjects = [os.path.basename(sub_path).replace("sub-", "") for sub_path in subject_paths]
        return subjects
    
    @staticmethod
    def subject_session_entity(subject_id, session_id=None):
        """
        Create a string representation of a subject and optional session entity.

        Parameters:
        subject_id (str): The label of the subject (e.g., '01').
        session_id (str, optional): The label of the session (e.g., '01'). Defaults to None.

        Returns:
        str: A string in the format 'sub-<subject_id>[_ses-<session_id>]'.
        """
        entity = f"sub-{subject_id}"
        if session_id is not None:
            entity += f"_ses-{session_id}"
        return entity