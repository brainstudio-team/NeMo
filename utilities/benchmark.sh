#!/bin/bash
cd $1
make && make install

EXAMPLE=./src/examples/mpi_example
CONFIG_FILE=$2
NEURONS_NUM=(100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000)
SYNAPSES_NUM=(500 1000)
SIM_STEPS=1000
MAPPER_TYPES=(0 1);
HOST_FILE=$3
LOG_DIR=$4

mtype=1

NUMBER_OF_MACHINES=(5 9 17);
for procs in ${NUMBER_OF_MACHINES[*]};
	do 

	echo "Number of processes: $procs"
	for ncount in ${NEURONS_NUM[*]}; 
	do

		for scount in ${SYNAPSES_NUM[*]}; 
		do
				echo "Starting simulation..."
				echo "  neurons: $ncount"
				echo "  synapses per neuron: $scount"
				echo

				mpiexec -np $procs --machinefile $HOST_FILE $EXAMPLE $ncount $scount $SIM_STEPS $mtype firings.txt $CONFIG_FILE $4
				echo

				mul=$(($ncount*$scount))
				echo "Simulation $i & $procs & $ncount & $mul \\\\" >> latextable.txt
		done
		echo
		echo " \hline" >> latextable.txt
	done
done

#gather logs into a single file
LOGDIRS=(`ls $LOG_DIR`)

cd $LOG_DIR
pwd
for curDir in ${LOGDIRS[*]};
do 
	echo $curDir
	cd $curDir

	for i in w*.txt
	do
		echo $i
		cat $i >> worker-stats-all.txt
	done
	cd ..
done
