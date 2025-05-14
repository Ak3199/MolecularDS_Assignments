#!/bin/bash
function echoMe {
        mpirun -np 1 /home/ccus/software/lmp_mpi < in_$1.lammps > dump_$1.log
        exit 0
}
