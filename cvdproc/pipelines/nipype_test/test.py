import os
from nipype import Node, Workflow
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec
from nipype.interfaces.utility import IdentityInterface
from traits.api import Str

class TestInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(mandatory=True, desc="Subject ID")
    session_id = Str(desc="Session ID")
    output_path = Str(desc="Output path")

class TestOutputSpec(TraitedSpec):
    sub_and_ses_txt = Str(desc="Txt file of Subject and Session IDs")

class Test(BaseInterface):
    input_spec = TestInputSpec
    output_spec = TestOutputSpec

    def _run_interface(self, runtime):
        subject_id = self.inputs.subject_id
        session_id = self.inputs.session_id
        
        sub_and_ses = f"Subject ID: {subject_id}, Session ID: {session_id}"

        # make sure the output_path exists
        os.makedirs(self.inputs.output_path, exist_ok=True)

        # save it to a text file
        with open(os.path.join(self.inputs.output_path, "sub_and_ses.txt"), "w") as f:
            f.write(sub_and_ses)

        self._sub_and_ses_txt = os.path.join(self.inputs.output_path, "sub_and_ses.txt")

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["sub_and_ses_txt"] = self._sub_and_ses_txt

        return outputs

class TestPipeline:
    def __init__(self, subject, session, output_path, **kwargs):
        """
        Test pipeline
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)
    
    def check_data_requirements(self):
        return True
    
    def create_workflow(self):
        # print the subject and session IDs
        test_workflow = Workflow(name='test_workflow')

        inputnode = Node(IdentityInterface(fields=["subject_id", "session_id", "output_path"]), name="inputnode")
        inputnode.inputs.subject_id = self.session.subject_id
        inputnode.inputs.session_id = self.session.session_id
        inputnode.inputs.output_path = self.output_path

        testnode = Node(Test(), name="testnode")

        test_workflow.connect(inputnode, "subject_id", testnode, "subject_id")
        test_workflow.connect(inputnode, "session_id", testnode, "session_id")
        test_workflow.connect(inputnode, "output_path", testnode, "output_path")

        return test_workflow
