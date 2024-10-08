#!/bin/bash

#SBATCH --job-name=cifar10_resnet
#SBATCH --output="./logs/resnets"
#SBATCH --nodelist=nv174
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "


echo "Run started at:- "
date

# ex) srun python -m mnist_resnet50.train

#srun python -m cifar10_resnet.train -model resnet20
#srun python -m cifar10_resnet.test -model resnet20
#srun python -m cifar10_resnet.train -model resnet32
#srun python -m cifar10_resnet.test -model resnet32
#srun python -m cifar10_resnet.train -model plain20
#srun python -m cifar10_resnet.test -model plain20
#srun python -m cifar10_resnet.train -model plain32
#srun python -m cifar10_resnet.test -model plain32
#srun python -m cifar10_resnet.train -model resnet44
#srun python -m cifar10_resnet.test -model resnet44
#srun python -m cifar10_resnet.train -model plain44
#srun python -m cifar10_resnet.test -model plain44
#srun python -m cifar10_resnet.train -model resnet56
#srun python -m cifar10_resnet.test -model resnet56
#srun python -m cifar10_resnet.train -model plain56
#srun python -m cifar10_resnet.test -model plain56
#srun python -m cifar10_resnet.train -model resnet110
#srun python -m cifar10_resnet.test -model resnet110
#srun python -m cifar10_resnet.train -model resnet1202
#srun python -m cifar10_resnet.test -model resnet1202

#cnt=1
#while [ 1 == 1 ]
#do
#  if [ $cnt -eq 5 ]; then
#    break
#  fi
#  srun python -m cifar10_resnet.train -model resnet56
#  srun python -m cifar10_resnet.test -model resnet56
#  let cnt++
#done
