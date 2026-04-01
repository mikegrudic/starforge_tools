#!/bin/bash
#SBATCH --job-name=SKIRT
#SBATCH --account=cca ##IF User is on Rusty
#SBATCH --partition=cca ##IF User is on Rusty
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --time=14:00:00
##SBATCH --mem=1400G
#SBATCH --chdir={working_dir}
#SBATCH --mail-type=ALL
#SBATCH --mail-user= {user}
#SBATCH --export=all

module purge
module load slurm gcc openmpi

## Load envioronement where skirt is compiled

SNAPSHOT="/mnt/ceph/users/starforge/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output/snapshot_1700.hdf5"

echo "Job started on $(hostname)"
echo "Working directory: $(pwd)"
echo "Snapshot: ${SNAPSHOT}"
echo "MPI processes: ${SLURM_NTASKS}"
echo "Threads per process: ${SLURM_CPUS_PER_TASK}"

python run_skirt.py "${SNAPSHOT}" \
    --output_path=skirt_output \
    --mpi_processes="${SLURM_NTASKS}" \
    --threads="${SLURM_CPUS_PER_TASK}"