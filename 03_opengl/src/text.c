#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"
#include "stdbool.h"
#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

#include "text.h"

static stbtt_bakedchar g_cdata[96]; // ASCII 32..126 is 95 glyphs
static GLuint g_ftex = 0;

bool InitTextRes( char * font ) {

	FILE* fp = fopen( font, "rb" );
	if ( !fp ) return false;
	fseek( fp, 0, SEEK_END );
	int size = ftell( fp );
	fseek( fp, 0, SEEK_SET );
	
	unsigned char* ttfBuffer = (unsigned char*)malloc( size ); 
	if ( !ttfBuffer ) {
		fclose( fp );
		return false;
	}
	
	fread(ttfBuffer, 1, size, fp);
	fclose(fp);
	fp = 0;
	
	unsigned char* bmap = (unsigned char*)malloc( 512 * 512 );

	if ( !bmap ) {
		free( ttfBuffer );
		return false;
	}
	
	stbtt_BakeFontBitmap( ttfBuffer, 0, 15.0f, bmap, 512, 512, 32, 96, g_cdata );
	
	glGenTextures( 1, &g_ftex );
	glBindTexture( GL_TEXTURE_2D, g_ftex );
	glTexImage2D( GL_TEXTURE_2D, 0, GL_ALPHA, 512, 512, 0, GL_ALPHA, GL_UNSIGNED_BYTE, bmap );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

	free( ttfBuffer );
	free( bmap );

	return true;

}

void DestroyTextRes() {
	if (g_ftex) {
		glDeleteTextures( 1, &g_ftex );
		g_ftex = 0;
	}
}

static void GetBakedQuad(stbtt_bakedchar *chardata, int pw, int ph, int char_index,
						 float *xpos, float *ypos, stbtt_aligned_quad *q) {

	stbtt_bakedchar *b = chardata + char_index;
	int round_x = STBTT_ifloor( *xpos + b->xoff );
	int round_y = STBTT_ifloor( *ypos - b->yoff );
	
	q->x0 = (float)round_x;
	q->y0 = (float)round_y;
	q->x1 = (float)round_x + b->x1 - b->x0;
	q->y1 = (float)round_y - b->y1 + b->y0;
	
	q->s0 = b->x0 / (float)pw;
	q->t0 = b->y0 / (float)pw;
	q->s1 = b->x1 / (float)ph;
	q->t1 = b->y1 / (float)ph;
	
	*xpos += b->xadvance;
}


static const float g_tabStops[4] = {150, 210, 270, 330};

static float GetTextLength(stbtt_bakedchar *chardata, const char* text)
{
	float xpos = 0;
	float len = 0;
	while (*text)
	{
		int c = (unsigned char)*text;
		if (c == '\t')
		{
			unsigned int i;
			for (i = 0; i < 4; ++i)
			{
				if (xpos < g_tabStops[i])
				{
					xpos = g_tabStops[i];
					break;
				}
			}
		}
		else if (c >= 32 && c < 128)
		{
			stbtt_bakedchar *b = chardata + c-32;
			int round_x = STBTT_ifloor((xpos + b->xoff) + 0.5);
			len = round_x + b->x1 - b->x0 + 0.5f;
			xpos += b->xadvance;
		}
		++text;
	}
	return len;
}

void DrawText( float x, float y, const char *text, int align, unsigned int col ) {
	if (!g_ftex) return;
	if (!text) return;
	
	if (align == TEXT_ALIGN_CENTER)
		x -= GetTextLength(g_cdata, text)/2;
	else if (align == TEXT_ALIGN_RIGHT)
		x -= GetTextLength(g_cdata, text);
	
	glColor4ub(col&0xff, (col>>8)&0xff, (col>>16)&0xff, (col>>24)&0xff);
	
	glEnable(GL_TEXTURE_2D);
	
	glBindTexture(GL_TEXTURE_2D, g_ftex);
	
	glBegin(GL_TRIANGLES);
	
	const float ox = x;
	
	while (*text)
	{
		int c = (unsigned char)*text;
		if (c == '\t')
		{
			unsigned int i;
			for (i = 0; i < 4; ++i)
			{
				if (x < g_tabStops[i]+ox)
				{
					x = g_tabStops[i]+ox;
					break;
				}
			}
		}
		else if (c >= 32 && c < 128)
		{			
			stbtt_aligned_quad q;

			GetBakedQuad(g_cdata, 512,512, c-32, &x,&y,&q);
			
			glTexCoord2f(q.s0, q.t0);
			glVertex2f(q.x0, q.y0);
			glTexCoord2f(q.s1, q.t1);
			glVertex2f(q.x1, q.y1);
			glTexCoord2f(q.s1, q.t0);
			glVertex2f(q.x1, q.y0);
			
			glTexCoord2f(q.s0, q.t0);
			glVertex2f(q.x0, q.y0);
			glTexCoord2f(q.s0, q.t1);
			glVertex2f(q.x0, q.y1);
			glTexCoord2f(q.s1, q.t1);
			glVertex2f(q.x1, q.y1);
		}
		++text;
	}
	
	glEnd();	
	glDisable(GL_TEXTURE_2D);
}

