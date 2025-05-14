#!/bin/bash


sol_per_pop=47

for i in $(seq 0 $sol_per_pop)
do
    cp jobsubmission.sh jobsubmission_${i}.sh
    echo "echoMe $i" >> jobsubmission_${i}.sh
    chmod +x jobsubmission_${i}.sh
    ./jobsubmission_${i}.sh
    sleep 15
done
