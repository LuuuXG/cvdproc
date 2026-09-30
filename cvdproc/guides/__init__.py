"""Install application guidance without initializing or modifying imaging data."""
from pathlib import Path


def install_application_guides(bids_dir):
    """Copy missing guidance files; preserve all existing dataset instructions."""
    root = Path(bids_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'BIDS directory must already exist: {root}')
    source = Path(__file__).parent / 'application'
    targets = [(source / 'dataset_agents.md', root / 'AGENTS.md')]
    targets.extend((path, root / 'code' / 'agent' / 'guides' / path.name)
                   for path in sorted(source.glob('*.md')) if path.name != 'dataset_agents.md')
    installed = []
    preserved = []
    for origin, destination in targets:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open('x', encoding='utf-8') as stream:
                stream.write(origin.read_text(encoding='utf-8'))
            installed.append(destination.name)
        except FileExistsError:
            preserved.append(destination.name)
    ignore_path = root / '.bidsignore'
    existing = ignore_path.read_text(encoding='utf-8') if ignore_path.exists() else ''
    if 'AGENTS.md' not in existing.splitlines():
        with ignore_path.open('a', encoding='utf-8') as stream:
            stream.write(('\n' if existing and not existing.endswith('\n') else '') + 'AGENTS.md\n')
    summary = []
    if installed:
        summary.append('installed ' + ', '.join(installed))
    if preserved:
        summary.append('kept existing ' + ', '.join(preserved))
    print('Agent instructions: ' + '; '.join(summary) + '.')


def show_agent_next_steps(bids_dir):
    """Show the application entry point after the requested operation finishes."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    root = Path(bids_dir).expanduser().resolve()
    message = Text(f'Open your agent in {root}, or set this folder as its project directory.\n\n'
                   'Ask it to read AGENTS.md and help you convert DICOMs, run an analysis, or extract results. '
                   'Tell it what you want to do; it will guide you through the inputs and settings.\n'
                   'Task guides are in code/agent/guides/.')
    Console().print(Panel(message, title='Next steps', border_style='bold cyan', style='bold cyan'))
