#ifndef data
#define data

#define DataFileSize 81920
#define NbData	1024
#define DataOffset DataFileSize/NbData
#define path "src/dubinski.tab"

int dataExtraction(float masse[], float posX[], float posY[], float posZ[], float mobX[], float mobY[], float mobZ[], int Size);

#endif
