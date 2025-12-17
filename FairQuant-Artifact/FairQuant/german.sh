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

if [ "$PA" == "age" ]; then
    idx=11 # 11 stands for age
elif [ "$PA" == "sex" ]; then
    idx=19 # 19 stands for sex
else
    echo "Error: invalid PA provided for german: age or sex"
    exit 1
fi

# for ((i=1; i<=5; i++)); do # for each model 1 to 7
#     echo -e "\n-----Running network GC-$i on $PA-----"
#     # ./network_test "../models/german/GC-$i.nnet" "$idx"
#     ./FairQuant-Artifact/FairQuant/network_test "FairQuant-Artifact/models/german/GC-$i.nnet" "$idx"
# done

./FairQuant-Artifact/FairQuant/network_test "FairQuant-Artifact/models/german/GC-10.nnet" "$idx"
