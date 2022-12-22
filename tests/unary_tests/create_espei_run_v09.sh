#!/bin/bash
#SBATCH --job-name="Copt4K"
#SBATCH --output="TestCoopMg.out"
#SBATCH --error="TestCoopMg.err"
#SBATCH --account=MELTCRUQ
#SBATCH --partition=bdwall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH -t 00:15:00

module purge
module load anaconda3/2019.10
#export PATH=/lcrc/project/PhaseDiagramsUQ/PyCalESPEI_Latest/pycalphad-espei_11_30_2021/bin:$PATH
export PATH=/lcrc/project/MeltCrUQ/NewEspeiMod/bin:$PATH
rm -rf dask-worker-space/ scheduler.json 

#python subs.py 
#rm Cu-Mg_ESPEI*.tdb
#mpirun -np 36 dask-mpi --nthreads 1 --no-nanny --scheduler-file scheduler.json --memory-limit 0 &
#python trunc.py
python run_mcmc.py
#python run_mcmc_restart.py
