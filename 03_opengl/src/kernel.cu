#include "cuda.h"
#include "kernel.cuh"
#include "stdio.h"

__global__ void kernel_calculate (particule * tab, particule_pos * tab_ret) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < NbData) {
		int j = 0;
		float acX = 0;
		float acY = 0;
		float acZ = 0;
		//printf("Dans Boucle 1 \n");
		for ( j = 0; j < NbData; j++ ) {
			float tempo_acc = 0; 
			temp tempo;
			//printf("Boucle 2 numéro %d \n", j);
			if (j != i) {	
		
				// Parrallel 1
				tempo.deltaX = tab[j].pos.posX - tab[i].pos.posX;
				tempo.deltaY = tab[j].pos.posY - tab[i].pos.posY;
				tempo.deltaZ = tab[j].pos.posZ - tab[i].pos.posZ;
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
			//tab[i].acc.accX = acX;
			//tab[i].acc.accY = acY;
			//tab[i].acc.accZ = acZ;
			
			// Parrallel 0
			tab[i].mob.mobX += acX;
			tab[i].pos.posX += tab[i].mob.mobX * 0.1f;
			tab_ret[i].posX = tab[i].pos.posX; 

			tab[i].mob.mobY += acY;
			tab[i].pos.posY += tab[i].mob.mobY * 0.1f;
			tab_ret[i].posY = tab[i].pos.posY; 

			tab[i].mob.mobZ += acZ;
			tab[i].pos.posZ += tab[i].mob.mobZ * 0.1f;
			tab_ret[i].posZ = tab[i].pos.posZ; 
			// Fin

	}

	return;
}

void CalculateMove_k (int NbBlock, int NbThread, particule * tab, particule_pos * tab_ret) {
	kernel_calculate<<<NbBlock,NbThread>>>(tab, tab_ret);
	return;
}
