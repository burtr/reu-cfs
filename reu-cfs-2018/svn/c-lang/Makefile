


all: helloworld one-d-array two-d-array
	@echo "Runnning Hello World ..."
	./helloworld
	@echo "Running one-d-array ... "
	./one-d-array
	@echo "Running two-d-array ... "
	./two-d-array
	make clean

helloworld: helloworld.c

one-d-array: one-d-array.c
	cc -o $@ $<

two-d-array: two-d-array.c
	cc -o $@ $<


clean:
	-rm helloworld one-d-array two-d-array

