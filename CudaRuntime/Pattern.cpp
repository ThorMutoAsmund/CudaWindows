#include "Pattern.h"

void pattern(int WIDTH, int HEIGHT, unsigned char* lpBitmapBits)
{
    int index = 0;
    for (int x = 0; x < HEIGHT; x++)
    {
        for (int y = 0; y < WIDTH; y++)
        {
            lpBitmapBits[index + 0] = 128;  // blue
            lpBitmapBits[index + 1] = x % 256; // green
            lpBitmapBits[index + 2] = y % 256;  // red 
            index += 4;
        }
    }
}
