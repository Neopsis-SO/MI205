#include <stdio.h>

int main( int argc, char ** argv ) {

	int i = 0;

 	printf( "hello world\n");
	printf( "now listing arguments...\n" );

	for ( i = 0; i < argc; i++ ) {
		printf("arg[%d] = %s\n", i, argv[ i ] );
	}

	return 1;
}

