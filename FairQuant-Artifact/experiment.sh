#!/usr/bin/env bash
ulimit -c unlimited
set -x  # Show each command as it runs
set -e  # Exit immediately if any command fails

echo "Current directory: $(pwd)"

# Download make and gcc
sudo apt update
sudo apt install -y build-essential
sudo apt install csvtool
sudo apt install -y python3-swiftclient
sudo apt install -y gdb

echo "Verifying installation..."
gcc --version
make --version

# Download the tar file
wget https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.6/OpenBLAS-0.3.6.tar.gz   
tar -xzf OpenBLAS-0.3.6.tar.gz

# Set up installation path
export INSTALL_PREFIX=$HOME/OpenBLAS
mkdir -p "$INSTALL_PREFIX"

# Install OpenBLAS
cd OpenBLAS-0.3.6
# make
make TARGET=GENERIC
make PREFIX="$INSTALL_PREFIX" install

# Set environment variables
export LIBRARY_PATH="$INSTALL_PREFIX/lib:$LIBRARY_PATH"
export C_INCLUDE_PATH="$INSTALL_PREFIX/include:$C_INCLUDE_PATH"
export LD_LIBRARY_PATH="$INSTALL_PREFIX/lib:$LD_LIBRARY_PATH"  

cd ..

echo "Current directory: $(pwd)"
ls -la FairQuant-Artifact/FairQuant || echo "Directory missing!"

# Build FairQuant
FAIRQUANT_DIR="$PWD/FairQuant-Artifact/FairQuant"
make -C "$FAIRQUANT_DIR" all
make -C "$FAIRQUANT_DIR" CFLAGS="-g -O0" all

############################################################

ADULT_SCRIPT="./FairQuant-Artifact/FairQuant/adult.sh"
if [ ! -f "$ADULT_SCRIPT" ]; then
    echo "ERROR: $ADULT_SCRIPT does not exist!" >&2
    exit 1
fi

if [ ! -x "$ADULT_SCRIPT" ]; then
    echo "ERROR: $ADULT_SCRIPT is not executable!" >&2
    chmod +x "$ADULT_SCRIPT"
fi

# Run adult.sh
echo "Running $ADULT_SCRIPT with argument 'sex'"
"$ADULT_SCRIPT" sex

############################################################

# BANK_SCRIPT="./FairQuant-Artifact/FairQuant/bank.sh"
# if [ ! -f "$BANK_SCRIPT" ]; then
#     echo "ERROR: $BANK_SCRIPT does not exist!" >&2
#     exit 1
# fi

# if [ ! -x "$BANK_SCRIPT" ]; then
#     echo "ERROR: $BANK_SCRIPT is not executable!" >&2
#     chmod +x "$BANK_SCRIPT"
# fi

# # Run adult.sh
# echo "Running $BANK_SCRIPT with argument 'age'"
# "$BANK_SCRIPT" age

############################################################

# GERMAN_SCRIPT="./FairQuant-Artifact/FairQuant/german.sh"
# if [ ! -f "$GERMAN_SCRIPT" ]; then
#     echo "ERROR: $GERMAN_SCRIPT does not exist!" >&2
#     exit 1
# fi

# if [ ! -x "$GERMAN_SCRIPT" ]; then
#     echo "ERROR: $GERMAN_SCRIPT is not executable!" >&2
#     chmod +x "$GERMAN_SCRIPT"
# fi

# # Run adult.sh
# echo "Running $GERMAN_SCRIPT with argument 'sex'"
# "$GERMAN_SCRIPT" age

############################################################

# COMPAS_SCRIPT="./FairQuant-Artifact/FairQuant/compas.sh"
# if [ ! -f "$COMPAS_SCRIPT" ]; then
#     echo "ERROR: $COMPAS_SCRIPT does not exist!" >&2
#     exit 1
# fi

# if [ ! -x "$COMPAS_SCRIPT" ]; then
#     echo "ERROR: $COMPAS_SCRIPT is not executable!" >&2
#     chmod +x "$COMPAS_SCRIPT"
# fi

# # Run adult.sh
# echo "Running $COMPAS_SCRIPT with argument 'sex'"
# "$COMPAS_SCRIPT" sex

############################################################

# DEFAULT_SCRIPT="./FairQuant-Artifact/FairQuant/default.sh"
# if [ ! -f "$DEFAULT_SCRIPT" ]; then
#     echo "ERROR: $DEFAULT_SCRIPT does not exist!" >&2
#     exit 1
# fi

# if [ ! -x "$DEFAULT_SCRIPT" ]; then
#     echo "ERROR: $DEFAULT_SCRIPT is not executable!" >&2
#     chmod +x "$DEFAULT_SCRIPT"
# fi

# # Run adult.sh
# echo "Running $DEFAULT_SCRIPT with argument 'SEX'"
# "$DEFAULT_SCRIPT" SEX

############################################################

# LSAC_SCRIPT="./FairQuant-Artifact/FairQuant/lsac.sh"
# if [ ! -f "$LSAC_SCRIPT" ]; then
#     echo "ERROR: $LSAC_SCRIPT does not exist!" >&2
#     exit 1
# fi

# if [ ! -x "$LSAC_SCRIPT" ]; then
#     echo "ERROR: $LSAC_SCRIPT is not executable!" >&2
#     chmod +x "$LSAC_SCRIPT"
# fi

# # Run adult.sh
# echo "Running $LSAC_SCRIPT with argument 'race'"
# "$LSAC_SCRIPT" race

############################################################

# UCI_SCRIPT="./FairQuant-Artifact/FairQuant/uci.sh"
# if [ ! -f "$UCI_SCRIPT" ]; then
#     echo "ERROR: $UCI_SCRIPTT does not exist!" >&2
#     exit 1
# fi

# if [ ! -x "$UCI_SCRIPT" ]; then
#     echo "ERROR: $UCI_SCRIPT is not executable!" >&2
#     chmod +x "$UCI_SCRIPT"
# fi

# # Run adult.sh
# echo "Running $UCI_SCRIPT with argument 'sex'"
# "$UCI_SCRIPT" sex

############################################################


# cc-generate-openrc
source ~/openrc

bucket_name="bare_metal_experiment_pattern_data"
file_to_upload="FairQuant-Artifact/FairQuant/counterexamples.csv"
object_name="counterexamples.csv"   # <-- THIS WAS MISSING

echo
echo "Uploading results to the object store container $bucket_name"

swift post "$bucket_name"

# Correctly delete the previous object
swift delete "$bucket_name" "$object_name" 2>/dev/null || true

if [ -f "$file_to_upload" ]; then
    echo "Uploading $file_to_upload"
    swift upload "$bucket_name" "$file_to_upload" --object-name "$object_name"
else
    echo "ERROR: File $file_to_upload does not exist!" >&2
    exit 1
fi