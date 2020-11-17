#include "cuda.h"
#include "kernel.cuh"
#include "stdio.h"

__global__ void kernel_calculate (particule * tab, particule_pos * tab_ret) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < NbData) {
	
		if (tab[i].galaxyName != 2) {
			tab_ret[i].posX = tab[i].pos.posX;
			tab_ret[i].posY = tab[i].pos.posY;
			tab_ret[i].posZ = tab[i].pos.posZ;
			tab[i].galaxyName = 2;
		}

		else {
			int j = 0;
			float acX = 0;
			float acY = 0;
			float acZ = 0;
			// On stock dans un registre les variables de pos pour faire les delta
			// Car L'accès à un registre (proche des coeurs) est plus rapide qu'un accès RAM
			// (Accès RAM = accès le plus long de tous, excepté les stockages externes)
		    float px = tab_ret[i].posX;
		    float py = tab_ret[i].posY;
		    float pz = tab_ret[i].posZ;

			for ( j = 0; j < NbData; j++ ) {
				float tempo_acc = 0; 
				temp tempo;

				if (j != i) {		
					tempo.deltaX = tab_ret[j].posX - px;
					tempo.deltaY = tab_ret[j].posY - py;
					tempo.deltaZ = tab_ret[j].posZ - pz;

					tempo.d_ij = sqrtf( tempo.deltaX*tempo.deltaX + tempo.deltaY*tempo.deltaY + tempo.deltaZ*tempo.deltaZ );

					if (tempo.d_ij < 1)
						tempo.d_ij = 1;

					tempo_acc = MASS_FACTOR*DAMPING_FACTOR*(1/(tempo.d_ij*tempo.d_ij*tempo.d_ij))*tab[j].masse; // Meme calcul realise trois fois

					acX += tempo.deltaX * tempo_acc;
					acY += tempo.deltaY * tempo_acc;
					acZ += tempo.deltaZ * tempo_acc;
				}		
			}
			tab[i].mob.mobX += acX;
			tab_ret[i].posX += tab[i].mob.mobX * 0.1f;

			tab[i].mob.mobY += acY;
			tab_ret[i].posY += tab[i].mob.mobY * 0.1f;

			tab[i].mob.mobZ += acZ;
			tab_ret[i].posZ += tab[i].mob.mobZ * 0.1f; 
		}
	}

	return;
}


void CalculateMove_k (int NbBlock, int NbThread, particule * tab, particule_pos * tab_ret) {
	kernel_calculate<<<NbBlock,NbThread>>>(tab, tab_ret);
	return;
}

