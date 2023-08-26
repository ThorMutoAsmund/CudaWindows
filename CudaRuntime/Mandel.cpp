#include "Mandel.h"

#define MAGNITUDE_CUTOFF 4
#define NUMCOLOURS 256


float  width_fact, height_fact;

int col(int x, int y, float xmin, float ymin);

void mandel(int WIDTH, int HEIGHT, float xmin, float xmax, float ymin, float ymax, unsigned char* lpBitmapBits)
{
    COLORREF color = RGB(255, 0, 0);

    width_fact = (xmax - xmin) / WIDTH;
    height_fact = (ymax - ymin) / HEIGHT;

    int index = 0;
    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            int iter = col(x, y, xmin, ymin);
            int blue = (iter & 0x0f) << 4;
            int green = (iter & 0xf0) << 0;
            int red = (iter & 0xf00) >> 4;

            lpBitmapBits[index + 0] = blue;  // blue
            lpBitmapBits[index + 1] = green; // green
            lpBitmapBits[index + 2] = red;  // red 
            index += 4;
        }
    }
}


int col(int x, int y, float xmin, float ymin)
{
    int n, icount = 0;
    float p, q, r, i, prev_r, prev_i;

    p = (((float)x) * width_fact) + (float)xmin;
    q = (((float)y) * height_fact) + (float)ymin;

    prev_i = 0;
    prev_r = 0;

    for (n = 0; n <= NUMCOLOURS; n++)
    {
        r = (prev_r * prev_r) - (prev_i * prev_i) + p;
        i = 2 * (prev_r * prev_i) + q;

        if (r * r + i * i >= MAGNITUDE_CUTOFF)
        {
            break;
        }

        prev_r = r;
        prev_i = i;
    }
    return n;
}