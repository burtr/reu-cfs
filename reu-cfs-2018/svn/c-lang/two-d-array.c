#include<stdio.h>
#include<string.h>
#include<stdlib.h>

int main(int argc, char * argv[]) {
	int i, j, k ;
	char a[3][5] ;
	char * cp ;

	printf("size of char: %lu\n", sizeof(char)) ;
	printf("size of array: %lu\n", sizeof(a)) ;

	k = 0 ;
	for (i=0;i<3;i++)
		for (j=0;j<5;j++) {
			printf("%p a[%d][%d] gets %d\n", &a[i][j], i, j, k) ;
			a[i][j] = k++ ;
	}

	// lay a vector over the array
	// note: in the following, a is of type sequence of sequence of char's,
	// where the first sequence has 3 elements, and the second 5. 
	// Indexing once, give a sequence of chars, and therefore the 
	// assignment evaluates to pointer to char.

	//  a[0]                     a[1]          ... a[2]
	//  a[][0] a[][1] ... a[][4] a[][0] a[][1] ... a[][0] ... a[][4]

	cp = a[0] ; 
	for (i=0;i<15;i++) {
		printf("%p cp[%d] has %d\n", &cp[i], i, cp[i]) ;
	}	
	return 0 ;
}

