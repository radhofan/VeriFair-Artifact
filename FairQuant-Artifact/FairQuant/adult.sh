#!/usr/bin/env bash
ulimit -c unlimited
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

if [ "$PA" == "sex" ]; then
    idx=8 # 8 stands for sex
else
    echo "Error: invalid PA provided for adult: sex"
    exit 1
fi

# # Just test the loop
# for ((i=1; i<=12; i++)); do
#     echo -e "\n-----Running network AC-$i on $PA-----"
#     model_path="./FairQuant-Artifact/models/adult/AC-$i.nnet"
#     echo "Model Path: $model_path"
# done

# for ((i=1; i<=12; i++)); do # for each model 1 to 12
#     echo -e "\n-----Running network AC-$i on $PA-----"
#     # ./network_test "../models/adult/AC-$i.nnet" "$idx"
#     ./FairQuant-Artifact/FairQuant/network_test "FairQuant-Artifact/models/adult/AC-$i.nnet" "$idx"
# done      

./FairQuant-Artifact/FairQuant/network_test "FairQuant-Artifact/models/adult/AC-2-Retrained.nnet" "$idx" 