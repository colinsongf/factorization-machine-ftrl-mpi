#!/bin/bash

#tt=`date`
#mkdir backup/"$tt"
#mv train backup/"$tt"
#mv *.log backup/"$tt"
#mv core backup/"$tt"
#make
#rm log/*
rocess_number=3
Ip=("10.101.2.89" "10.101.2.90")
for ip in ${Ip[@]}
do
    ssh worker@$ip rm /home/worker/xiaoshu/factorization-machine-ftrl-mpi/train
done
scp train worker@10.101.2.89:/home/worker/xiaoshu/factorization-machine-ftrl-mpi/.
scp train worker@10.101.2.90:/home/worker/xiaoshu/factorization-machine-ftrl-mpi/.
mpirun -f ./hosts -np $process_number ./train ftrl 5 100 ./data/agaricus.txt.train ./data/agaricus.txt.test
