#!/bin/sh

# declare variables to use as commands
source ../src/rheokit_commands.sh

# create .str files (for Tinker-HP simulations)
$shear --list=HPoutputs.txt

# gather .str files, run integral for each file,
# take average and std deviation for fit.
python $gkintegrate --parquet EtOH_hp.parquet \
	--temp 323.15 --box-edge 47.0 --start-idx 500000 \
	--cutoff-lag 2000000 --stress-freq 1 \
	--manifest stress_manifest.txt \
	--log-name 01_gkintegrate.log


# check averaged curve for flatness
python $platchk --parquet EtOH_hp.parquet \
	--cutoff 200 300 400 500 600 700 750 800 900 1000 \
	--log-name 02_platchk.log

# perform fitting procedure
python $stressfit --parquet EtOH_hp.parquet \
	--cutoff 750 --log-name 03_stressfit.log \
       	--json-name EtOH_fit_params.json


## NOTE: Tinker25 generates .str files by default.
## The stress_manifest.txt file needs to be created manually.
## Do this with a find command, for example:
##
## find /path/to/search -name "*.str" > stress_manifest.txt
