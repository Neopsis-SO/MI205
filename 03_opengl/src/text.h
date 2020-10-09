#ifndef __TEXT_H__
#define __TEXT_H__

#define RGBA(r,g,b,a) (r) | (g << 8) | (b << 16) | (a << 24)

#define TEXT_ALIGN_LEFT 0
#define TEXT_ALIGN_RIGHT 1
#define TEXT_ALIGN_CENTER 2

bool InitTextRes( char * font );
void DestroyTextRes();
void DrawText( float x, float y, const char *text, int align, unsigned int col );

#endif

