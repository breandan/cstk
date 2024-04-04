#!/bin/bash
#SBATCH --time 24:0:0
#SBATCH --account=def-jinguo
#SBATCH --nodes=1
#SBATCH --output=/scratch/b/bengioy/breandan/slurm-%j.out
#SBATCH --error=/scratch/b/bengioy/breandan/slurm-%j.err
#SBATCH --mail-user=bre@ndan.co
#SBATCH --mail-type=ALL

# export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2017/Core/cudacore/10.2.89/targets/x86_64-linux/lib/
# export TRANSFORMERS_OFFLINE=1
# module load python/3.8
# Load Java through CCEnv when running on Niagara:
module load CCEnv StdEnv/2020 java/17.0.2
# module load java/17.0.2
# source bin/activate
# ./gradlew --offline completeCode | tee logfile.txt

commit_message=$(git log -1 --pretty=format:"%s" | sed 's/ /_/g')
java -Xmx150G -jar gym-fs-fat-1.0-SNAPSHOT.jar 2>&1 | tee /scratch/b/bengioy/breandan/log_${commit_message}.txt