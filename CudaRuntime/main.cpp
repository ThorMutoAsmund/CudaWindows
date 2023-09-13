#include "main.h"

using namespace std;

#define WIDTH 1024
#define HEIGHT 1024

// Tutorials
// http://www.winprog.org/tutorial/bitmaps.html

const char g_szClassName[] = "myWindowClass";
void initDIBS(HWND hwnd);
void render(HWND hwnd);

BITMAPINFO bi;
HBITMAP bitmap;
HDC hDCMem;
unsigned char* lpBitmapBits;
double xmin = -2.10, xmax = 0.85, ymin = -1.5, ymax = 1.5;
int zoom = 0;
bool turbo = false;
int xPos;
int yPos;
bool last = true;
bool ready = false;

CudaArgs cpuArgs;

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_CREATE:
    {
        initDIBS(hwnd);
        render(hwnd);
        SetTimer(hwnd, 1, 1000/100, NULL);
    }
    break;

    case WM_TIMER:
    {
        if ((zoom != 0 || last) && ready)
        {
            if (zoom)
            {
                double fac = turbo ? 0.9 : 0.96;
                fac = zoom < 0 ? 1.0 / fac : fac;
                double newWidth = (xmax - xmin) * fac;
                double newHeight = (ymax - ymin) * fac;

                double c = (double)xPos / WIDTH;
                xmin = xmin * (1.0 - c) + xmax * c - (c * newWidth);
                xmax = xmin + newWidth;

                c = (double)yPos / HEIGHT;
                ymin = ymin * (1.0 - c) + ymax * c - (c * newHeight);
                ymax = ymin + newHeight;
            }
 
            ready = false;
            render(hwnd);

            InvalidateRect(hwnd, NULL, true);

            last = false;
        }
    }
    break;

    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        BITMAP bm;
        auto hbmOld = SelectObject(hDCMem, bitmap);

        GetObject(bitmap, sizeof(bm), &bm);
        BitBlt(hdc, 0, 0, bm.bmWidth, bm.bmHeight, hDCMem, 0, 0, SRCCOPY);

        EndPaint(hwnd, &ps);
        ready = true;
    }
    break;

    case WM_LBUTTONDOWN:
    case WM_RBUTTONDOWN:
    {
        zoom = msg == WM_LBUTTONDOWN ? 1 : -1;
    }
    break;

    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
    {
        zoom = 0;
        last = true;
    }
    break;

    case WM_MOUSEMOVE:
    {
        xPos = GET_X_LPARAM(lParam);
        yPos = GET_Y_LPARAM(lParam);
    }
    break;

    case WM_KEYDOWN:
    {
        if (wParam == 0x41)
        {
            cpuArgs.aa = cpuArgs.aa == 0 ? 1 : (cpuArgs.aa == 1 ? 2 : 0);
            last = true;
            render(hwnd);
            last = false;
            InvalidateRect(hwnd, NULL, true);
        }
        else if (wParam == 0x42)
        {
            cpuArgs.slow = 1 - cpuArgs.slow;
            last = true;
            render(hwnd);
            last = false;
            InvalidateRect(hwnd, NULL, true);
        }
        else if (wParam == 0x49)
        {
            if (turbo && cpuArgs.iterations > 256)
            {
                cpuArgs.iterations -= 256;
            }
            else if (!turbo)
            {
                cpuArgs.iterations += 256;
            }
            last = true;
            render(hwnd);
            last = false;
            InvalidateRect(hwnd, NULL, true);
        }
        else if (wParam == VK_SHIFT)
        {
            turbo = true;
        }
        break;
    }
    break;

    case WM_KEYUP:
    {
        if (wParam == VK_SHIFT)
        {
            turbo = false;
        }
        break;
    }
    break;

    case WM_CLOSE:
    {
        DeleteDC(hDCMem);
        DeleteObject(bitmap);
        DestroyWindow(hwnd);
    }
    break;

    case WM_DESTROY:
    {
        //DeleteDC(hDCMem);
        DeleteObject(bitmap);
        PostQuitMessage(0);
    }
    break;

    default:
        return DefWindowProc(hwnd, msg, wParam, lParam);
    }
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
    LPSTR lpCmdLine, int nCmdShow)
{
    cpuArgs.scrheight = HEIGHT;
    cpuArgs.scrwidth = WIDTH;
    cpuArgs.aa = 2;
    cpuArgs.iterations = 512;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int cores = getSPcores(prop);

    char debug[256];
    sprintf(debug, "Cores: %d (major %d) (minor %d)\n", cores, prop.major, prop.minor);
    OutputDebugString(debug);


    // Create bitmap info
    ZeroMemory(&bi, sizeof(BITMAPINFO));
    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = WIDTH;
    bi.bmiHeader.biHeight = -HEIGHT;  //negative so (0,0) is at top left
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;

    // Register the Window Class
    WNDCLASSEX wc;
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = 0;
    wc.lpfnWndProc = WndProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = hInstance;
    wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszMenuName = NULL;
    wc.lpszClassName = g_szClassName;
    wc.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

    if (!RegisterClassEx(&wc))
    {
        MessageBox(NULL, "Window Registration Failed!", "Error!", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    // Create and show the Window
    HWND hwnd = CreateWindowEx(
        WS_EX_CLIENTEDGE,
        g_szClassName,
        "CUDA test",
        WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME,
        CW_USEDEFAULT, CW_USEDEFAULT, WIDTH, 43 + HEIGHT,
        NULL, NULL, hInstance, NULL);

    if (hwnd == NULL)
    {
        MessageBox(NULL, "Window Creation Failed!", "Error!", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    // The Message Loop
    MSG Msg;
    while (GetMessage(&Msg, NULL, 0, 0) > 0)
    {
        TranslateMessage(&Msg);
        DispatchMessage(&Msg);
    }

    return Msg.wParam;
}

void initDIBS(HWND hwnd)
{
    HDC hdc = GetDC(hwnd);
    hDCMem = CreateCompatibleDC(hdc);
    bitmap = ::CreateDIBSection(hDCMem, &bi, DIB_RGB_COLORS, (VOID**)&lpBitmapBits, NULL, 0);

    //memset(lpBitmapBits, 0, WIDTH * HEIGHT * sizeof(long));

    //DeleteDC(hDCMem);
    ReleaseDC(hwnd, hdc);
}

void render(HWND hwnd)
{
    cpuArgs.width = (xmax - xmin);
    cpuArgs.height = (ymax - ymin);
    cpuArgs.xmin = xmin;
    cpuArgs.ymin = ymin;
    cpuArgs.last = last ? 1 : 0;

    cudaMandel(&cpuArgs, lpBitmapBits);
}


