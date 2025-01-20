#!/bin/bash

GPU='#SBATCH --gpus=1'
CONSTRAINT='#SBATCH --gres=gpumem:50g'

RUN_NAME="mib_$MODEL"

run_ids=( 0 1 2 3 4 )


for var in "${run_ids[@]}"
do

COMMAND="python attribute_all.py --run_id $var"

echo $COMMAND

sbatch <<EOT
#!/bin/bash
#SBATCH -n 4
$GPU
$CONSTRAINT
#SBATCH --output="/cluster/home/stolfoa/bsub_logs/mib/attribute_all/run_$var"
#SBATCH --job-name=$RUN_NAME
#SBATCH --mem-per-cpu=16000
#SBATCH -t 23:59:00

# Your commands here
module load eth_proxy

$COMMAND

EOT

done