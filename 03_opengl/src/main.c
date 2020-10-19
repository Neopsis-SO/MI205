#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>

#include "GL/glew.h"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

#include "text.h"

#define DataFileSize 81920
#define NbData	1024
#define DataOffset DataFileSize/NbData
#define path "src/dubinski.tab"

#define DISK_SIZE 16384	//Fisrt in database
#define BULGE_SIZE 8192	//Second in database
#define HALO_SIZE 16384	//Third in database

#define MASS_FACTOR 10
#define DAMPING_FACTOR 1

static float g_inertia = 0.5f;

static float oldCamPos[] = { 0.0f, 0.0f, -45.0f };
static float oldCamRot[] = { 0.0f, 0.0f, 0.0f };
static float newCamPos[] = { 0.0f, 0.0f, -45.0f };
static float newCamRot[] = { 0.0f, 0.0f, 0.0f };

static bool g_showGrid = true;
static bool g_showAxes = true;

/*------------------------------*/
// Variables for simulation 

struct particule {
	int galaxyName; //0:Milky Way / 1:Andromeda
	float masse;
	float posX;
	float posY;
	float posZ;
	float mobX;
	float mobY;
	float mobZ;
	float accX;
	float accY;
	float accZ;
};

struct temp {
	float deltaX;
	float deltaY;
	float deltaZ;
	float d_ij;
};

struct particule all_particules[NbData];
/*------------------------------*/

void DrawGridXZ( float ox, float oy, float oz, int w, int h, float sz ) {
	
	int i;

	glLineWidth( 1.0f );

	glBegin( GL_LINES );

	glColor3f( 0.48f, 0.48f, 0.48f );

	for ( i = 0; i <= h; ++i ) {
		glVertex3f( ox, oy, oz + i * sz );
		glVertex3f( ox + w * sz, oy, oz + i * sz );
	}

	for ( i = 0; i <= h; ++i ) {
		glVertex3f( ox + i * sz, oy, oz );
		glVertex3f( ox + i * sz, oy, oz + h * sz );
	}

	glEnd();

}

int dataExtraction(struct particule data[])
{
	float buffMasse = 0.0f;
	float buffPosX = 0.0f;
	float buffPosY = 0.0f;
	float buffPosZ = 0.0f;
	float buffMobX = 0.0f;
	float buffMobY = 0.0f;
	float buffMobZ = 0.0f;

	int offsetIndex = 0;
	int lineIndex = 0;
	int dataIndex = 0;

	FILE * DataFile = fopen(path, "r");
	if (path == NULL) {
		return -1;
	}	

	for (lineIndex = 0; lineIndex < DataFileSize; lineIndex++) {
		fscanf(DataFile, "%f %f %f %f %f %f %f\n", &buffMasse, &buffPosX, &buffPosY, &buffPosZ, &buffMobX, &buffMobY, &buffMobZ);

		if(offsetIndex++ == DataOffset-1) {
			if(lineIndex < DISK_SIZE)
				data[dataIndex].galaxyName = 0;
			else if(lineIndex < DISK_SIZE*2)
				data[dataIndex].galaxyName = 1;
			else if(lineIndex < ((BULGE_SIZE)+DISK_SIZE*2))
				data[dataIndex].galaxyName = 0;
			else if(lineIndex < ((BULGE_SIZE+DISK_SIZE)*2))
				data[dataIndex].galaxyName = 1;
			else if(lineIndex < ((HALO_SIZE)+BULGE_SIZE*2+DISK_SIZE*2))
				data[dataIndex].galaxyName = 0;
			else
				data[dataIndex].galaxyName = 1;
			
			offsetIndex = 0;
			data[dataIndex].masse = buffMasse;
			data[dataIndex].posX = buffPosX;
			data[dataIndex].posY = buffPosY;
			data[dataIndex].posZ = buffPosZ;
			data[dataIndex].mobX = buffMobX;
			data[dataIndex].mobY = buffMobY;
			data[dataIndex].mobZ = buffMobZ;
			dataIndex++;
		}
	}
	fclose(DataFile);
	printf("data extraction complited\n");
	return 0;
}

void CalculateMove (struct particule tab[]) {
	int i = 0;
	int j = 0;
	
	// Loop to calculate all new pos
	#pragma omp parallel for
	for (i = 0; i < NbData; i++) {
		// Parrallel 0
		tab[i].mobX += tab[i].accX;
		tab[i].accX = 0;
		tab[i].posX += tab[i].mobX * 0.1;

		tab[i].mobY += tab[i].accY;
		tab[i].accY = 0;
		tab[i].posY += tab[i].mobY * 0.1;

		tab[i].mobZ += tab[i].accZ;
		tab[i].accZ = 0;
		tab[i].posZ += tab[i].mobZ * 0.1;
		// Fin
	}

//Bonne méthode en simple core, mauvaise en multi core
/*	#pragma omp parallel for
	for (i = 0; i < NbData; i++) {
		
		float acX = 0;
		float acY = 0;
		float acZ = 0;
		#pragma omp parallel for reduction (+:acX,acY,acZ)
		for (j = i+1; j < NbData; j++) {
			float tempo_acc = 0; //
			struct temp tempo;

			tempo.deltaX = tab[j].posX - tab[i].posX;
			tempo.deltaY = tab[j].posY - tab[i].posY;
			tempo.deltaZ = tab[j].posZ - tab[i].posZ;
			// Pas de pow() car appel trop lourd pour calcul
			tempo.d_ij = sqrtf( tempo.deltaX*tempo.deltaX + tempo.deltaY*tempo.deltaY + tempo.deltaZ*tempo.deltaZ );
			// Calcul slide 31
			if (tempo.d_ij < 1)
				tempo.d_ij = 1;
			tempo_acc = MASS_FACTOR*DAMPING_FACTOR*(1/(tempo.d_ij*tempo.d_ij*tempo.d_ij))*tab[j].masse; // Meme calcul realise trois fois

			acX += tempo.deltaX * tempo_acc;
			acY += tempo.deltaY * tempo_acc;
			acZ += tempo.deltaZ * tempo_acc;

			tab[j].accX += -tempo.deltaX * tempo_acc;
			tab[j].accY += -tempo.deltaY * tempo_acc;
			tab[j].accZ += -tempo.deltaZ * tempo_acc;
		}
		tab[i].accX += acX;
		tab[i].accY += acY;
		tab[i].accZ += acZ;
		// On remarque que les calculs sont réalisés deux fois, au signe prêt (exemple, si i = 3 et j = 5, on fait
		// le même calcul , au signe prêt, que lorsque i = 5 et j = 3/ En
		
	}*/
//float accX = 0, accY = 0, accZ = 0;
	
	for ( i = 0; i < NbData; i++ ) {
		//float tempo_acc = 0; 
		//struct temp tempo;
		float acX = 0;
		float acY = 0;
		float acZ = 0;
		//#pragma omp parallel for reduction (+:acX,acY,acZ)
		for ( j = 0; j < NbData; j++ ) {
			float tempo_acc = 0; 
			struct temp tempo;
			if (j != i) {	
			
				// Parrallel 1
				tempo.deltaX = tab[j].posX - tab[i].posX;
				tempo.deltaY = tab[j].posY - tab[i].posY;
				tempo.deltaZ = tab[j].posZ - tab[i].posZ;
				// Fin parrallel 1

				tempo.d_ij = sqrtf( tempo.deltaX*tempo.deltaX + tempo.deltaY*tempo.deltaY + tempo.deltaZ*tempo.deltaZ );

				if (tempo.d_ij < 1)
					tempo.d_ij = 1;

				tempo_acc = MASS_FACTOR*DAMPING_FACTOR*(1/(tempo.d_ij*tempo.d_ij*tempo.d_ij))*tab[j].masse; // Meme calcul realise trois fois

				// Parrallel 2
				acX += tempo.deltaX * tempo_acc;
				acY += tempo.deltaY * tempo_acc;
				acZ += tempo.deltaZ * tempo_acc;
				// Fin Parrallel 2
			}
		}
		tab[i].accX = acX;
		tab[i].accY = acY;
		tab[i].accZ = acZ;
	}

	return;
}



void DrawGalaxies (struct particule aff[])
{
	int i;

	glPointSize( 0.1f );
	glBegin( GL_POINTS );

	for ( i = 0; i <= NbData; i++ ) {
		if(!aff[i].galaxyName) {
			glColor3f(0.449f, 0.758f, 0.980f);	//(115/256, 194/256, 251/256) 
		} else {
			glColor3f(0.934f, 0.605f, 0.059f);	//(239/256, 155/256, 15/256) 
		}
		glVertex3f( aff[i].posX, aff[i].posY, aff[i].posZ);
	}
	glEnd();
}

void ShowAxes() {

	glLineWidth( 2.0f );

	glBegin( GL_LINES );
	
	glColor3f( 1.0f, 0.0f, 0.0f );
	glVertex3f( 0.0f, 0.0f, 0.0f );
	glVertex3f( 2.0f, 0.0f, 0.0f );

	glColor3f( 0.0f, 1.0f, 0.0f );
	glVertex3f( 0.0f, 0.0f, 0.0f );
	glVertex3f( 0.0f, 2.0f, 0.0f );

	glColor3f( 0.0f, 0.0f, 1.0f );
	glVertex3f( 0.0f, 0.0f, 0.0f );
	glVertex3f( 0.0f, 0.0f, 2.0f );
	
	glEnd();
}

int main( int argc, char ** argv ) 
{
	SDL_Event event;
	SDL_Window * window;
	SDL_DisplayMode current;
  	
	int width = 640;
	int height = 480;

	bool done = false;

	float mouseOriginX = 0.0f;
	float mouseOriginY = 0.0f;

	float mouseMoveX = 0.0f;
	float mouseMoveY = 0.0f;

	float mouseDeltaX = 0.0f;
	float mouseDeltaY = 0.0f;

	struct timeval begin, end;
	float fps = 0.0;
	char sfps[40] = "FPS: ";

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
										width, height, 
										SDL_WINDOW_OPENGL );
  
	SDL_GLContext glWindow = SDL_GL_CreateContext( window );

	GLenum status = glewInit();

	if ( status != GLEW_OK ) {
		printf( "error: unable to init glew\n" );
		return -1;
	}

	if ( ! InitTextRes( "./bin/DroidSans.ttf" ) ) {
		printf( "error: unable to init text resources\n" );
		return -1;
	}

	SDL_GL_SetSwapInterval( 1 );

	/**********************/
	
	dataExtraction(all_particules);

	/**********************/

	while ( !done ) {
  		
		int i;

		while ( SDL_PollEvent( &event ) ) {
      
			unsigned int e = event.type;
			
			if ( e == SDL_MOUSEMOTION ) {
				mouseMoveX = event.motion.x;
				mouseMoveY = height - event.motion.y - 1;
			} else if ( e == SDL_KEYDOWN ) {
				if ( event.key.keysym.sym == SDLK_F1 ) {
					g_showGrid = !g_showGrid;
				} else if ( event.key.keysym.sym == SDLK_F2 ) {
					g_showAxes = !g_showAxes;
				} else if ( event.key.keysym.sym == SDLK_ESCAPE ) {
 					done = true;
				}
			}

			if ( e == SDL_QUIT ) {
				printf( "quit\n" );
				done = true;
			}

		}

		mouseDeltaX = mouseMoveX - mouseOriginX;
		mouseDeltaY = mouseMoveY - mouseOriginY;

		if ( SDL_GetMouseState( 0, 0 ) & SDL_BUTTON_LMASK ) {
			oldCamRot[ 0 ] += -mouseDeltaY / 5.0f;
			oldCamRot[ 1 ] += mouseDeltaX / 5.0f;
		} else if ( SDL_GetMouseState( 0, 0 ) & SDL_BUTTON_RMASK ) {
			oldCamPos[ 2 ] += ( mouseDeltaY / 100.0f ) * 0.5 * fabs( oldCamPos[ 2 ] );
			oldCamPos[ 2 ]  = oldCamPos[ 2 ] > -5.0f ? -5.0f : oldCamPos[ 2 ];
		}

		mouseOriginX = mouseMoveX;
		mouseOriginY = mouseMoveY;

		glViewport( 0, 0, width, height );
		glClearColor( 0.2f, 0.2f, 0.2f, 1.0f );
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
		glEnable( GL_BLEND );
		glBlendEquation( GL_FUNC_ADD );
		glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
		glDisable( GL_TEXTURE_2D );
		glEnable( GL_DEPTH_TEST );
		glMatrixMode( GL_PROJECTION );
		glLoadIdentity();
		gluPerspective( 50.0f, (float)width / (float)height, 0.1f, 100000.0f );
		glMatrixMode( GL_MODELVIEW );
		glLoadIdentity();

		for ( i = 0; i < 3; ++i ) {
			newCamPos[ i ] += ( oldCamPos[ i ] - newCamPos[ i ] ) * g_inertia;
			newCamRot[ i ] += ( oldCamRot[ i ] - newCamRot[ i ] ) * g_inertia;
		}

		glTranslatef( newCamPos[0], newCamPos[1], newCamPos[2] );
		glRotatef( newCamRot[0], 1.0f, 0.0f, 0.0f );
		glRotatef( newCamRot[1], 0.0f, 1.0f, 0.0f );
		
		if ( g_showGrid ) {
			DrawGridXZ( -100.0f, 0.0f, -100.0f, 20, 20, 10.0 );
		}

		if ( g_showAxes ) {
			ShowAxes();
		}

		gettimeofday( &begin, NULL );

		/*------------------------------*/
		// Simulation should be computed here
		CalculateMove(all_particules);
		DrawGalaxies(all_particules);
		 
		/*------------------------------*/

		gettimeofday( &end, NULL );

		fps = (float)1.0f / ( ( end.tv_sec - begin.tv_sec ) * 1000000.0f + end.tv_usec - begin.tv_usec) * 1000000.0f;
		sprintf( sfps, "FPS : %.4f", fps );

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluOrtho2D(0, width, 0, height);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		DrawText( 10, height - 20, sfps, TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, 30, "'F1' : show/hide grid", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, 10, "'F2' : show/hide axes", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
	
		SDL_GL_SwapWindow( window );
		SDL_UpdateWindowSurface( window );
	}

	SDL_GL_DeleteContext( glWindow );
	DestroyTextRes();
	SDL_DestroyWindow( window );
	SDL_Quit();
	return 1;
}

