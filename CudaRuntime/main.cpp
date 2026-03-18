#include "main.h"
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

const int kRenderSizeMin = 512;
const int kRenderSizeMax = 2048;
const int kRenderSizeStep = 128;
int renderSize = kRenderSizeMax;

// Tutorials
// http://www.winprog.org/tutorial/bitmaps.html

const char g_szClassName[] = "myWindowClass";
void initDIBS(HWND hwnd);
void setRenderSize(HWND hwnd, int newSize);
void render(HWND hwnd);

BITMAPINFO bi;
HBITMAP bitmap;
HDC hDCMem;
HBITMAP frameBitmap;
HDC hDCFrame;
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
bool panning = false;
int panLastX = 0;
int panLastY = 0;
bool last = true;
bool ready = false;
int overlayMode = 0; // 0=full, 1=compact, 2=off
const double kIterationExponent = 1.3;
double iterationScale = 16.0;

CudaArgs cpuArgs;

unsigned int hsvToRgb(float h, float s, float v)
{
    h = h - floorf(h);
    const float hf = h * 6.0f;
    const int sector = (int)floorf(hf);
    const float f = hf - sector;

    const float p = v * (1.0f - s);
    const float q = v * (1.0f - s * f);
    const float t = v * (1.0f - s * (1.0f - f));

    float r = 0.0f, g = 0.0f, b = 0.0f;
    switch (sector % 6)
    {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    default: r = v; g = p; b = q; break;
    }

    return
        ((unsigned int)(r * 255.0f) << 16) |
        ((unsigned int)(g * 255.0f) << 8) |
        (unsigned int)(b * 255.0f);
}

unsigned int lerpRgb(unsigned int c0, unsigned int c1, float t)
{
    const float r0 = (float)((c0 >> 16) & 0xFF);
    const float g0 = (float)((c0 >> 8) & 0xFF);
    const float b0 = (float)(c0 & 0xFF);
    const float r1 = (float)((c1 >> 16) & 0xFF);
    const float g1 = (float)((c1 >> 8) & 0xFF);
    const float b1 = (float)(c1 & 0xFF);

    const unsigned int r = (unsigned int)roundf(r0 + (r1 - r0) * t);
    const unsigned int g = (unsigned int)roundf(g0 + (g1 - g0) * t);
    const unsigned int b = (unsigned int)roundf(b0 + (b1 - b0) * t);

    return (r << 16) | (g << 8) | b;
}

void generateRandomPalette(unsigned int palette[128])
{
    unsigned int anchors[4];
    const float baseHue = (float)rand() / (float)RAND_MAX;
    const float hueStep = 0.10f + 0.30f * ((float)rand() / (float)RAND_MAX);

    for (int i = 0; i < 4; ++i)
    {
        const float hueJitter = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.10f;
        const float h = baseHue + hueStep * i + hueJitter;
        const float s = 0.65f + 0.35f * ((float)rand() / (float)RAND_MAX);
        const float v = 0.45f + 0.55f * ((float)rand() / (float)RAND_MAX);
        anchors[i] = hsvToRgb(h, s, v);
    }

    for (int i = 0; i < 128; ++i)
    {
        const int seg = i / 32;               // 0..3
        const int segNext = (seg + 1) % 4;    // wrap last segment to first anchor
        const float t = (float)(i % 32) / 31.0f;
        palette[i] = lerpRgb(anchors[seg], anchors[segNext], t);
    }
}

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
    const char* antialiasState = cpuArgs.aa ? "ON" : "OFF";

    char overlay[512];
    if (overlayMode == 0)
    {
        sprintf(
            overlay,
            "Use L/R mouse button to zoom, and shift for turbo. Press scroll wheel to pan.\nF1=Info toggle, A=AntiAlias, C=Color cycle, I=Iteration multiplier, R=Screen resolution\nCenter: (%.12f, %.12f)\nZoom: %.3fx\nIterations: %d\nScale: %.1f\nAntialias: %s",
            centerX,
            centerY,
            zoomFactor,
            cpuArgs.iterations,
            iterationScale,
            antialiasState);
    }
    else
    {
        sprintf(
            overlay,
            "Center: (%.12f, %.12f)\nZoom: %.3fx\nIterations: %d\nScale: %.1f\nAntialias: %s",
            centerX,
            centerY,
            zoomFactor,
            cpuArgs.iterations,
            iterationScale,
            antialiasState);
    }

    RECT rc = { 10, 10, renderSize - 10, 220 };
    SetBkMode(hdc, TRANSPARENT);

    // Draw a thicker black outline for stronger readability.
    SetTextColor(hdc, RGB(0, 0, 0));
    for (int oy = -2; oy <= 2; ++oy)
    {
        for (int ox = -2; ox <= 2; ++ox)
        {
            if (ox == 0 && oy == 0)
            {
                continue;
            }
            RECT outline = { rc.left + ox, rc.top + oy, rc.right + ox, rc.bottom + oy };
            DrawTextA(hdc, overlay, -1, &outline, DT_LEFT | DT_TOP | DT_NOPREFIX);
        }
    }

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
        const bool panMoved = panning && (xPos != panLastX || yPos != panLastY);
        if ((zoom != 0 || panMoved || last) && ready)
        {
            if (zoom)
            {
                double fac = turbo ? 0.9 : 0.96;
                fac = zoom < 0 ? 1.0 / fac : fac;
                double newWidth = (xmax - xmin) * fac;
                double newHeight = (ymax - ymin) * fac;

                double c = (double)xPos / renderSize;
                xmin = xmin * (1.0 - c) + xmax * c - (c * newWidth);
                xmax = xmin + newWidth;

                c = (double)yPos / renderSize;
                ymin = ymin * (1.0 - c) + ymax * c - (c * newHeight);
                ymax = ymin + newHeight;
            }
            else if (panMoved)
            {
                const int dx = xPos - panLastX;
                const int dy = yPos - panLastY;
                const double width = xmax - xmin;
                const double height = ymax - ymin;

                const double moveX = -(double)dx * width / renderSize;
                const double moveY = -(double)dy * height / renderSize;

                xmin += moveX;
                xmax += moveX;
                ymin += moveY;
                ymax += moveY;

                panLastX = xPos;
                panLastY = yPos;
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
        auto hbmOldSrc = SelectObject(hDCMem, bitmap);
        auto hbmOldDst = SelectObject(hDCFrame, frameBitmap);

        GetObject(bitmap, sizeof(bm), &bm);
        BitBlt(hDCFrame, 0, 0, bm.bmWidth, bm.bmHeight, hDCMem, 0, 0, SRCCOPY);
        if (overlayMode != 2)
        {
            drawOverlay(hDCFrame);
        }
        BitBlt(hdc, 0, 0, bm.bmWidth, bm.bmHeight, hDCFrame, 0, 0, SRCCOPY);

        SelectObject(hDCMem, hbmOldSrc);
        SelectObject(hDCFrame, hbmOldDst);

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

    case WM_MBUTTONDOWN:
    {
        panning = true;
        panLastX = GET_X_LPARAM(lParam);
        panLastY = GET_Y_LPARAM(lParam);
        xPos = panLastX;
        yPos = panLastY;
        SetCapture(hwnd);
    }
    break;

    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
    {
        zoom = 0;
        last = true;
    }
    break;

    case WM_MBUTTONUP:
    {
        if (panning)
        {
            panning = false;
            ReleaseCapture();
            last = true;
        }
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
            overlayMode = (overlayMode + 1) % 3;
            InvalidateRect(hwnd, NULL, false);
        }
        else if (wParam == 0x41)
        {
            cpuArgs.aa = cpuArgs.aa ? 0 : 1;
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
        else if (wParam == 0x43)
        {
            generateRandomPalette(cpuArgs.palette);
            cpuArgs.useCustomPalette = 1;
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
        else if (wParam == 0x52)
        {
            const bool shiftHeld = (GetKeyState(VK_SHIFT) & 0x8000) != 0;
            const int delta = shiftHeld ? -kRenderSizeStep : kRenderSizeStep;
            int newSize = renderSize + delta;
            if (newSize < kRenderSizeMin)
            {
                newSize = kRenderSizeMin;
            }
            else if (newSize > kRenderSizeMax)
            {
                newSize = kRenderSizeMax;
            }
            setRenderSize(hwnd, newSize);
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
        if (hDCMem) { DeleteDC(hDCMem); hDCMem = NULL; }
        if (hDCFrame) { DeleteDC(hDCFrame); hDCFrame = NULL; }
        if (bitmap) { DeleteObject(bitmap); bitmap = NULL; }
        if (frameBitmap) { DeleteObject(frameBitmap); frameBitmap = NULL; }
        DestroyWindow(hwnd);
    }
    break;

    case WM_DESTROY:
    {
        if (hDCMem) { DeleteDC(hDCMem); hDCMem = NULL; }
        if (hDCFrame) { DeleteDC(hDCFrame); hDCFrame = NULL; }
        if (bitmap) { DeleteObject(bitmap); bitmap = NULL; }
        if (frameBitmap) { DeleteObject(frameBitmap); frameBitmap = NULL; }
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
    srand((unsigned int)time(NULL));

    cpuArgs.scrheight = renderSize;
    cpuArgs.scrwidth = renderSize;
    cpuArgs.aa = 1;
    cpuArgs.useCustomPalette = 0;
    cpuArgs.iterations = getIterationsForZoom(getCurrentZoomFactor());
    xPos = renderSize / 2;
    yPos = renderSize / 2;
    panLastX = xPos;
    panLastY = yPos;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int cores = getSPcores(prop);

    char debug[256];
    sprintf(debug, "Cores: %d (major %d) (minor %d)\n", cores, prop.major, prop.minor);
    OutputDebugString(debug);


    // Create bitmap info
    ZeroMemory(&bi, sizeof(BITMAPINFO));
    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = renderSize;
    bi.bmiHeader.biHeight = -renderSize;  //negative so (0,0) is at top left
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
    RECT wr = { 0, 0, renderSize, renderSize };
    AdjustWindowRectEx(&wr, WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME, FALSE, WS_EX_CLIENTEDGE);

    HWND hwnd = CreateWindowEx(
        WS_EX_CLIENTEDGE,
        g_szClassName,
        "CUDA test",
        WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME,
        CW_USEDEFAULT, CW_USEDEFAULT, wr.right - wr.left, wr.bottom - wr.top,
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
    if (!hDCMem)
    {
        hDCMem = CreateCompatibleDC(hdc);
    }
    if (!hDCFrame)
    {
        hDCFrame = CreateCompatibleDC(hdc);
    }
    if (bitmap)
    {
        DeleteObject(bitmap);
        bitmap = NULL;
    }
    if (frameBitmap)
    {
        DeleteObject(frameBitmap);
        frameBitmap = NULL;
    }

    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = renderSize;
    bi.bmiHeader.biHeight = -renderSize;  //negative so (0,0) is at top left
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;

    bitmap = ::CreateDIBSection(hDCMem, &bi, DIB_RGB_COLORS, (VOID**)&lpBitmapBits, NULL, 0);
    frameBitmap = CreateCompatibleBitmap(hdc, renderSize, renderSize);

    //memset(lpBitmapBits, 0, renderSize * renderSize * sizeof(long));

    //DeleteDC(hDCMem);
    ReleaseDC(hwnd, hdc);
}

void setRenderSize(HWND hwnd, int newSize)
{
    if (newSize == renderSize)
    {
        return;
    }

    renderSize = newSize;
    cpuArgs.scrwidth = renderSize;
    cpuArgs.scrheight = renderSize;
    xPos = renderSize / 2;
    yPos = renderSize / 2;
    panLastX = xPos;
    panLastY = yPos;

    initDIBS(hwnd);

    RECT wr = { 0, 0, renderSize, renderSize };
    AdjustWindowRectEx(&wr, WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME, FALSE, WS_EX_CLIENTEDGE);
    SetWindowPos(
        hwnd,
        NULL,
        0,
        0,
        wr.right - wr.left,
        wr.bottom - wr.top,
        SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);

    last = true;
    ready = false;
    render(hwnd);
    InvalidateRect(hwnd, NULL, false);
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


