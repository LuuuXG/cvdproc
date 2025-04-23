# input_for_pipeline.py
from ...bids_data.subject import BIDSSubject
from ...bids_data.session import BIDSSession

# as a function
def pipeline_input(bids_dir, subject_id, session_id, output_path):
    from cvdproc.bids_data.subject import BIDSSubject
    from cvdproc.bids_data.session import BIDSSession

    subject = BIDSSubject(subject_id, bids_dir)

    session = next((s for s in subject.get_all_sessions() if s.session_id == session_id), None) if session_id else None

    return subject, session, output_path