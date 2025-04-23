import os
import glob
from .session import BIDSSession

class BIDSSubject:
    def __init__(self, subject_id, bids_dir):
        self.subject_id = subject_id
        self.bids_dir = os.path.abspath(bids_dir)
        self.subject_dir = os.path.join(self.bids_dir, f"sub-{subject_id}")

        if not os.path.exists(self.subject_dir):
            raise FileNotFoundError(f"Subject directory {self.subject_dir} does not exist.")

        self.sessions = self._find_sessions()
        self.sessions_id = self._find_sessions_id()

    def _find_sessions(self):
        sessions = []
        session_dirs = sorted(glob.glob(os.path.join(self.subject_dir, 'ses-*')))
        for session_dir in session_dirs:
            session_id = os.path.basename(session_dir).split('-')[1]
            sessions.append(BIDSSession(self.bids_dir, self.subject_id, session_id))
        return sessions
    
    def _find_sessions_id(self):
        session_dirs = sorted(glob.glob(os.path.join(self.subject_dir, 'ses-*')))
        session_ids = []
        for session_dir in session_dirs:
            session_id = os.path.basename(session_dir).split('-')[1]
            session_ids.append(session_id)
        return session_ids

    def get_all_sessions(self):
        return self.sessions
