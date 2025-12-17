#!/usr/bin/env bash

set -e
set -x

exec > >(tee -a FairQuant-Artifact/FairQuant/experiment_output.log)
exec 2>&1

# Check if PA is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [Protected Attribute]"
    exit 1
fi

PA="$1"
idx=-1

if [ "$PA" == "SEX" ]; then
    idx=1 #
else
    echo "Error: invalid PA provided"
    exit 1
fi

# for ((i=1; i<=8; i++)); do # for each model 1 to 8
#     echo -e "\n-----Running network BM-$i on $PA-----"
#     # ./network_test "../models/bank/BM-$i.nnet" "$idx"
#     ./FairQuant-Artifact/FairQuant/network_test "FairQuant-Artifact/models/bank/BM-$i.nnet" "$idx"
# done

 ./FairQuant-Artifact/FairQuant/network_test "FairQuant-Artifact/models/default/DF-1.nnet" "$idx"
