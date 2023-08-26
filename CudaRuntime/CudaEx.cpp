#include "CudaEx.h"

int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
    case 2: // Fermi
        if (devProp.minor == 1) cores = mp * 48;
        else cores = mp * 32;
        break;
    case 3: // Kepler
        cores = mp * 192;
        break;
    case 5: // Maxwell
        cores = mp * 128;
        break;
    case 6: // Pascal
        if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
        else if (devProp.minor == 0) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    case 7: // Volta and Turing
        if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    case 8: // Ampere (My gfx board 8 minor 6)
        if (devProp.minor == 0) cores = mp * 64;
        else if (devProp.minor == 6) cores = mp * 128;
        else if (devProp.minor == 9) cores = mp * 128; // ada lovelace
        else printf("Unknown device type\n");
        break;
    case 9: // Hopper
        if (devProp.minor == 0) cores = mp * 128;
        else printf("Unknown device type\n");
        break;
    default:
        printf("Unknown device type\n");
        break;
    }

    return cores;
}
