def run_matlab_script(script):
    import subprocess
    subprocess.run(f'matlab -nodisplay -nosplash -nodesktop -r "run(\'{script}\')"; exit;', shell=True)

    return script

if __name__ == '__main__':
    script = '/mnt/f/BIDS/demo_BIDS/derivatives/sepia_qsm/sub-YCHC0001/ses-01/sepia_qsm_script.m'
    run_matlab_script(script)