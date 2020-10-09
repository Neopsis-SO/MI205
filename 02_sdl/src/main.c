#include <stdio.h>
#include <stdbool.h>

#include "GL/glew.h"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

int main( int argc, char ** argv ) {

	SDL_Event event;
	SDL_Window * window;
	SDL_DisplayMode current;
  
	bool done = false;

	if ( SDL_Init ( SDL_INIT_EVERYTHING ) < 0 ) {
		printf( "error: unable to init sdl\n" );
		return -1;
	}

	if ( SDL_GetDesktopDisplayMode( 0, &current ) ) {
		printf( "error: unable to get current display mode\n" );
		return -1;
	}

	window = SDL_CreateWindow( "SDL", 	SDL_WINDOWPOS_CENTERED, 
										SDL_WINDOWPOS_CENTERED, 
										640, 480, 
										SDL_WINDOW_OPENGL );
  
	SDL_GLContext glWindow = SDL_GL_CreateContext( window );

	GLenum status = glewInit();

	if ( status != GLEW_OK ) {
		printf( "error: unable to init glew\n" );
		return -1;
	}

	SDL_GL_SetSwapInterval( 1 );

	while ( !done ) {
  
		while ( SDL_PollEvent( &event ) ) {
      
			unsigned int e = event.type;

			if ( e == SDL_KEYDOWN ) {
				if ( event.key.keysym.sym == SDLK_UP ) {
					printf( "up key pressed\n" );
				} else if ( event.key.keysym.sym == SDLK_DOWN ) {
					printf( "up down pressed\n" );
				} else if ( event.key.keysym.sym == SDLK_LEFT ) {
					printf( "up left pressed\n" );
				} else if ( event.key.keysym.sym == SDLK_RIGHT ) {
					printf( "up right pressed\n" );
				} else if ( event.key.keysym.sym == SDLK_ESCAPE ) {
 					done = true;
				}
			}

			if ( e == SDL_QUIT ) {
				printf( "quit\n" );
				done = true;
			}

		}
  
		SDL_GL_SwapWindow( window );
		SDL_UpdateWindowSurface( window );
	}

	SDL_GL_DeleteContext( glWindow );
	SDL_DestroyWindow( window );
	SDL_Quit();

	return 1;
}


