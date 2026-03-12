#!/bin/bash

args=("$@")

max_tries=5
success=0

for ((i=1; i<=max_tries; i++)); do
    echo ">>> Attempt $i running DRBUDDI_cuda..."
    DRBUDDI_cuda "${args[@]}"
    status=$?

    if [ $status -eq 0 ]; then
        echo ">>> DRBUDDI_cuda succeeded on attempt $i"
        success=1
        break
    else
        echo ">>> DRBUDDI_cuda failed (exit code $status), retrying..."
    fi
done

if [ $success -eq 0 ]; then
    echo ">>> DRBUDDI_cuda failed after $max_tries attempts, switching to CPU version..."

    cpu_args=()
    replace_done=0

    for ((i=0; i<${#args[@]}; i++)); do
        if [[ "${args[$i]}" == "--ncores" ]]; then
            cpu_args+=("--ncores" "8")
            ((i++))
            replace_done=1
        else
            cpu_args+=("${args[$i]}")
        fi
    done

    if [ $replace_done -eq 0 ]; then
        cpu_args+=("--ncores" "8")
    fi

    DRBUDDI "${cpu_args[@]}"
    status=$?

    if [ $status -eq 0 ]; then
        echo ">>> DRBUDDI (CPU) succeeded"
    else
        echo ">>> DRBUDDI (CPU) failed (exit code $status)"
        exit $status
    fi
fi