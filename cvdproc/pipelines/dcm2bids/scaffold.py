"""Run upstream scaffolding with a nonfatal software update check."""
import logging


def main():
    from dcm2bids.cli import dcm2bids_scaffold

    check_latest = dcm2bids_scaffold.check_latest

    def optional_update_check(*args, **kwargs):
        try:
            return check_latest(*args, **kwargs)
        except Exception as exc:
            logging.getLogger(__name__).warning('Software update check failed; continuing initialization: %s', exc)

    dcm2bids_scaffold.check_latest = optional_update_check
    try:
        return dcm2bids_scaffold.main()
    finally:
        dcm2bids_scaffold.check_latest = check_latest


if __name__ == '__main__':
    main()
