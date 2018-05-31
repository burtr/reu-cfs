#include<stdio.h>
#include<string.h>
#include<stdlib.h>

int main(int argc, char * argv[]) {
	int i ;
	char c[10] ;
	char * cp ;

	printf("size of char: %lu\n", sizeof(char)) ;
	printf("size of array: %lu\n", sizeof(c)) ;

	for (i=0;i<sizeof(c)/sizeof(c[0]);i++) {
		c[i] = i ;
	}
	for (cp=c; cp<c+10;cp++) {
		printf("%p %d\n", cp, (int)*cp ) ;
	}
	return 0 ;
}

