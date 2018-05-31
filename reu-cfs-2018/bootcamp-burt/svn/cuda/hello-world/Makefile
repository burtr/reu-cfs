
CUDA8= /usr/local/cuda-8.0
NVCC = ${CUDA8}/bin/nvcc

HW= hello-world

# test run:
# $ make run
# ./hello
# Hello World from CPU
# Hello World from GPU
# Hello World from GPU
# Hello World from GPU
# Hello World from GPU
# Hello World from GPU
# Hello World from GPU
# Hello World from GPU
# Hello World from GPU
# Hello World from GPU
# Hello World from GPU
# $

all:
	${NVCC} -arch sm_30 ${HW}.cu -o ${HW}

run:
	./${HW}

clean:
	-rm ${HW}
