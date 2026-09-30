"""Configure command-line interface output for CVDProc pipelines."""
from nipype.interfaces.base import CommandLine

# Apply before constructing interfaces, including those created in worker processes.
CommandLine.set_default_terminal_output("allatonce")
