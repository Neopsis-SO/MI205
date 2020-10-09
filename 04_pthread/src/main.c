#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include "thread.h"

int Compute( void * arg ) {
	int i;
	for ( i = 0; i < 10; i++ ) {
		printf( "doh!\n" );
	}
}

int main( int argc, char ** argv ) {

	bool done = false;

	thread_t * thread = NewThread();

	StartWorkerThread( thread, "compute", Compute, NULL, CORE_ANY, THREAD_NORMAL, DEFAULT_THREAD_STACK_SIZE );

	SignalWork( thread );
	
	//WaitForThread( thread ); 	/* blocking wait */
	
	while ( !done ) {
		if ( WorkIsDone( thread ) ) {
			done = true;
		}
	}

	return 1;
}

