P= dot-prod


$P: $P.cu
	nvcc -arch sm_30 -o $P $P.cu

run: $P
	./$P

clean:
	-rm $P
