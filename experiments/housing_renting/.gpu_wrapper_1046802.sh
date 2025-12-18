#!/bin/bash
# Set GPU based on local MPI rank
export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
echo "Rank $OMPI_COMM_WORLD_RANK (local $OMPI_COMM_WORLD_LOCAL_RANK) using GPU $CUDA_VISIBLE_DEVICES"
exec python3 "$@"
