"""Install dataset application instructions from the installed CVDProc package."""
import argparse

from . import install_application_guides, show_agent_next_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bids_dir', required=True, help='Existing BIDS root directory; existing instructions are preserved')
    args = parser.parse_args()
    install_application_guides(args.bids_dir)
    show_agent_next_steps(args.bids_dir)


if __name__ == '__main__':
    main()
