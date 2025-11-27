clear;

spm_path = '/this/is/for/nipype/spm_path';
pved_path = '/this/is/for/nipype/pved_path';

addpath(spm_path);
addpath(pved_path);

fib_dir = '/this/is/for/nipype/fib_dir';
pved_est(fib_dir, pved_path);