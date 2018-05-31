P= vector-add

$P: $P.cu 
	nvcc -arch sm_30 -lm -o $P $P.cu 

run: $P
	./$P

clean:
	-rm $P 
