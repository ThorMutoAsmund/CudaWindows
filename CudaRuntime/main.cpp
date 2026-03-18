#include "main.h"
#include <cmath>

using namespace std;

#define WIDTH 2048
#define HEIGHT 2048

// Tutorials
// http://www.winprog.org/tutorial/bitmaps.html

const char g_szClassName[] = "myWindowClass";
void initDIBS(HWND hwnd);
void render(HWND hwnd);

BITMAPINFO bi;
HBITMAP bitmap;
HDC hDCMem;
unsigned char* lpBitmapBits;
const double kInitialXMin = -2.10;
const double kInitialXMax = 0.85;
const double kInitialYMin = -1.5;
const double kInitialYMax = 1.5;
double xmin = kInitialXMin, xmax = kInitialXMax, ymin = kInitialYMin, ymax = kInitialYMax;
int zoom = 0;
bool turbo = false;
int xPos;
int yPos;
bool last = true;
bool ready = false;
bool showOverlay = true;
const double kIterationExponent = 1.3;
double iterationScale = 20.0;

CudaArgs cpuArgs;

double getCurrentZoomFactor()
{
    const double initialWidth = (kInitialXMax - kInitialXMin);
    const double currentWidth = (xmax - xmin);
    if (currentWidth <= 0.0)
    {
        return 1.0;
    }
    return initialWidth / currentWidth;
}

int getIterationsForZoom(double zoomFactor)
{
    // Smooth single-formula approximation:
    // iter ~= 100 + scale * ln(zoom)^exponent
    const double z = zoomFactor > 1.0 ? zoomFactor : 1.0;
    const double iter = 100.0 + iterationScale * pow(log(z), kIterationExponent);
    const int rounded = (int)round(iter);
    if (rounded < 100)
    {
        return 100;
    }
    if (rounded > 4096)
    {
        return 4096;
    }
    return rounded;
}

void drawOverlay(HDC hdc)
{
    const double centerX = (xmin + xmax) * 0.5;
    const double centerY = (ymin + ymax) * 0.5;
    const double zoomFactor = getCurrentZoomFactor();

    char overlay[256];
    sprintf(
        overlay,
        "Center: (%.12f, %.12f)\nZoom: %.3fx\nIterations: %d\nExponent: %.2f\nScale: %.1f",
        centerX,
        centerY,
        zoomFactor,
        cpuArgs.iterations,
        kIterationExponent,
        iterationScale);

    RECT rc = { 10, 10, WIDTH - 10, 180 };
    SetBkMode(hdc, TRANSPARENT);

    // Draw a black shadow for readability.
    SetTextColor(hdc, RGB(0, 0, 0));
    RECT shadow = { rc.left + 1, rc.top + 1, rc.right + 1, rc.bottom + 1 };
    DrawTextA(hdc, overlay, -1, &shadow, DT_LEFT | DT_TOP | DT_NOPREFIX);

    SetTextColor(hdc, RGB(255, 255, 255));
    DrawTextA(hdc, overlay, -1, &rc, DT_LEFT | DT_TOP | DT_NOPREFIX);
}

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

            InvalidateRect(hwnd, NULL, false);

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
        if (showOverlay)
        {
            drawOverlay(hdc);
        }

        EndPaint(hwnd, &ps);
        ready = true;
    }
    break;

    case WM_ERASEBKGND:
    {
        // Prevent default class brush erase to avoid white flash between frames.
        return 1;
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
        if (wParam == VK_F1)
        {
            showOverlay = !showOverlay;
            InvalidateRect(hwnd, NULL, false);
        }
        else if (wParam == 0x41)
        {
            cpuArgs.aa = cpuArgs.aa == 0 ? 1 : (cpuArgs.aa == 1 ? 2 : 0);
            last = true;
            render(hwnd);
            last = false;
            InvalidateRect(hwnd, NULL, false);
        }
        else if (wParam == 0x42)
        {
            cpuArgs.slow = 1 - cpuArgs.slow;
            last = true;
            render(hwnd);
            last = false;
            InvalidateRect(hwnd, NULL, false);
        }
        else if (wParam == 0x49)
        {
            const bool shiftHeld = (GetKeyState(VK_SHIFT) & 0x8000) != 0;
            iterationScale += shiftHeld ? -1.0 : 1.0;
            if (iterationScale < 1.0)
            {
                iterationScale = 1.0;
            }
            else if (iterationScale > 200.0)
            {
                iterationScale = 200.0;
            }
            iterationScale = round(iterationScale * 10.0) / 10.0;
            last = true;
            render(hwnd);
            last = false;
            InvalidateRect(hwnd, NULL, false);
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
    cpuArgs.iterations = getIterationsForZoom(getCurrentZoomFactor());

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
    wc.hbrBackground = NULL;
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
    cpuArgs.iterations = getIterationsForZoom(getCurrentZoomFactor());
    cpuArgs.width = (xmax - xmin);
    cpuArgs.height = (ymax - ymin);
    cpuArgs.xmin = xmin;
    cpuArgs.ymin = ymin;
    cpuArgs.last = last ? 1 : 0;

    cudaMandel(&cpuArgs, lpBitmapBits);
}


