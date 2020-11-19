#ifndef __MAIN_CUH__
#define __MAIN_CUH__

#define DataFileSize 81920
#define NbData	1024
#define DataOffset DataFileSize/NbData
#define path "src/dubinski.tab"

#define DISK_SIZE 16384	//Fisrt in database
#define BULGE_SIZE 8192	//Second in database
#define HALO_SIZE 16384	//Third in database

#define MASS_FACTOR 10
#define DAMPING_FACTOR 1

typedef struct {
	float posX;
	float posY;
	float posZ;
} particule_pos;

typedef struct {
	float mobX;
	float mobY;
	float mobZ;
} particule_mob;

typedef struct {
	float accX;
	float accY;
	float accZ;
} particule_acc;

typedef struct {
	int galaxyName; //0:Milky Way / 1:Andromeda
	float masse;
	particule_pos pos;
	particule_mob mob;
	particule_acc acc;
} particule;

typedef struct {
	float deltaX;
	float deltaY;
	float deltaZ;
	float d_ij;
} temp;

#endif