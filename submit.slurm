#!/bin/bash

#SBATCH --partition=cpu
#SBATCH -n 1
#SBATCH --array=1-10
#SBATCH --job-name=dynamics
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" params.txt`
parArray=($parameters)

tauhH=${parArray[0]}
tauthetaH=${parArray[1]}
DH=${parArray[2]}
JeHtoWM=${parArray[3]}
ITI=${parArray[4]}
sim=${parArray[5]}

echo "tauhH = $tauhH"
echo "tauthetaH = $tauthetaH"
echo "DH = $DH"
echo "JeHtoWM = $JeHtoWM"
echo "ITI = $ITI"
echo "sim = $sim"

python -u dynamics.py  $tauhH $tauthetaH $DH $JeHtoWM $ITI $sim
