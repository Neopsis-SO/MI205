#include <stdio.h>
#include "data.h"

/*
float masse[NbData] = {};
float posX[NbData] = {};
float posY[NbData] = {};
float posZ[NbData] = {};
float mobX[NbData] = {};
float mobY[NbData] = {};
float mobZ[NbData] = {};
*/

int dataExtraction(float masse[], float posX[], float posY[], float posZ[], float mobX[], float mobY[], float mobZ[], int Size) {
	float buffMasse;
	float buffPosX;
	float buffPosY;
	float buffPosZ;
	float buffMobX;
	float buffMobY;
	float buffMobZ;

	FILE * DataFile = fopen(path, "r");
	if (path == NULL) {
		return -1;
	}
	
	int i = 0, j = 0;

	for (i = 0; i < Size; i++) {
		fscanf(DataFile, "%f %f %f %f %f %f %f\n", &buffMasse, &buffPosX, &buffPosY, &buffPosZ, &buffMobX, &buffMobY, &buffMobZ);

		if(j++ == DataOffset-1) {
			j = 0;
			masse[(i+1)/DataOffset-1] = buffMasse;
			posX[(i+1)/DataOffset-1] = buffPosX;
			posY[(i+1)/DataOffset-1] = buffPosY;
			posZ[(i+1)/DataOffset-1] = buffPosZ;
			mobX[(i+1)/DataOffset-1] = buffMobX;
			mobY[(i+1)/DataOffset-1] = buffMobY;
			mobZ[(i+1)/DataOffset-1] = buffMobZ;
			//printf("%1.10f %f \n", masse[(i+1)/DataOffset-1], posX[(i+1)/DataOffset-1]);
		}
	}

	fclose(DataFile);

	return 0; //TODO: return number of data
}
