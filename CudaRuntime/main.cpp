#include "main.h"
#include <cmath>
#include <commdlg.h>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <gdiplus.h>
#include <string>
#include <vector>

using namespace std;
using namespace Gdiplus;

#pragma comment(lib, "gdiplus.lib")

const int kRenderSizeMin = 512;
const int kRenderSizeMax = 2048;
const int kRenderSizeStep = 128;
int renderWidth = 2048 - 3 * 256;
int renderHeight = ((2048 - 3 * 256) * 3) / 4;

// Tutorials
// http://www.winprog.org/tutorial/bitmaps.html

const char g_szClassName[] = "myWindowClass";
void initDIBS(HWND hwnd);
void setRenderWidth(HWND hwnd, int newWidth);
void saveImage(HWND hwnd);
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
char gpuName[256] = "Unknown";
int gpuCudaCores = 0;
double gpuCoreClockMHz = 0.0;
double gpuMemoryClockMHz = 0.0;
double gpuMemoryBandwidthGBs = 0.0;

CudaArgs cpuArgs;

int getEncoderClsid(const WCHAR* mimeType, CLSID* pClsid)
{
    UINT num = 0;
    UINT size = 0;
    GetImageEncodersSize(&num, &size);
    if (size == 0)
    {
        return -1;
    }

    std::vector<BYTE> buffer(size);
    ImageCodecInfo* encoders = (ImageCodecInfo*)buffer.data();
    if (GetImageEncoders(num, size, encoders) != Ok)
    {
        return -1;
    }

    for (UINT i = 0; i < num; ++i)
    {
        if (wcscmp(encoders[i].MimeType, mimeType) == 0)
        {
            *pClsid = encoders[i].Clsid;
            return (int)i;
        }
    }
    return -1;
}

std::wstring toWideString(const char* text)
{
    if (!text)
    {
        return std::wstring();
    }
    int len = MultiByteToWideChar(CP_ACP, 0, text, -1, nullptr, 0);
    if (len <= 0)
    {
        return std::wstring();
    }

    std::wstring out((size_t)len, L'\0');
    MultiByteToWideChar(CP_ACP, 0, text, -1, &out[0], len);
    out.resize((size_t)len - 1);
    return out;
}

bool setAsciiExifProperty(Bitmap& image, PROPID id, const char* text)
{
    PropertyItem item = {};
    item.id = id;
    item.type = 2; // ASCII
    item.length = (ULONG)strlen(text) + 1;
    item.value = (void*)text;
    return image.SetPropertyItem(&item) == Ok;
}

bool setUserCommentExifProperty(Bitmap& image, const char* text)
{
    static const BYTE prefix[] = { 'A', 'S', 'C', 'I', 'I', 0, 0, 0 };
    const size_t textLen = strlen(text);
    std::vector<BYTE> data(sizeof(prefix) + textLen + 1);
    memcpy(data.data(), prefix, sizeof(prefix));
    memcpy(data.data() + sizeof(prefix), text, textLen + 1);

    PropertyItem item = {};
    item.id = 0x9286; // Exif UserComment
    item.type = 7;    // UNDEFINED
    item.length = (ULONG)data.size();
    item.value = data.data();
    return image.SetPropertyItem(&item) == Ok;
}

void syncViewAspectToRenderSize()
{
    if (renderWidth <= 0 || renderHeight <= 0)
    {
        return;
    }

    const double centerY = (ymin + ymax) * 0.5;
    const double desiredHeight = (xmax - xmin) * ((double)renderHeight / (double)renderWidth);
    ymin = centerY - desiredHeight * 0.5;
    ymax = centerY + desiredHeight * 0.5;
}

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
    const char* smoothColorState = cpuArgs.smoothColoring ? "ON" : "OFF";

    char overlay[1024];
    if (overlayMode == 0)
    {
        sprintf(
            overlay,
            "Use L/R mouse button to zoom, and shift for turbo. Press scroll wheel to pan.\nF1=Info toggle, F2=Save, A=Antialias, C=Color cycle, I/Shift I=Iteration multiplier, R/Shift R=Screen resolution, O=Reset view, S=Smooth coloring\nCenter: (%.12f, %.12f)\nZoom: %.3fx\nIterations: %d\nMultiplier: %.1f\nAntialias: %s\nSmooth color: %s\nGPU: %s\nCUDA cores: %d\nCore clock: %.0f MHz\nMemory clock: %.0f MHz\nMemory bandwidth: %.1f GB/s",
            centerX,
            centerY,
            zoomFactor,
            cpuArgs.iterations,
            iterationScale,
            antialiasState,
            smoothColorState,
            gpuName,
            gpuCudaCores,
            gpuCoreClockMHz,
            gpuMemoryClockMHz,
            gpuMemoryBandwidthGBs);
    }
    else
    {
        sprintf(
            overlay,
            "Center: (%.12f, %.12f)\nZoom: %.3fx\nIterations: %d",
            centerX,
            centerY,
            zoomFactor,
            cpuArgs.iterations);
    }

    RECT rc = { 10, 10, renderWidth - 10, 340 };
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

void saveImage(HWND hwnd)
{
    const int saveWidth = 2048;
    int saveHeight = (int)round((double)saveWidth * (double)renderHeight / (double)renderWidth);
    if (saveHeight < 1)
    {
        saveHeight = 1;
    }

    time_t now = time(NULL);
    tm localTm;
    localtime_s(&localTm, &now);

    char defaultName[128];
    sprintf_s(
        defaultName,
        "mandelbrot_%04d%02d%02d_%02d%02d%02d_%dx%d.jpg",
        localTm.tm_year + 1900,
        localTm.tm_mon + 1,
        localTm.tm_mday,
        localTm.tm_hour,
        localTm.tm_min,
        localTm.tm_sec,
        saveWidth,
        saveHeight);

    char filename[MAX_PATH] = {};
    strncpy_s(filename, defaultName, _TRUNCATE);

    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = hwnd;
    ofn.lpstrFilter = "JPEG Files (*.jpg;*.jpeg)\0*.jpg;*.jpeg\0Bitmap Files (*.bmp)\0*.bmp\0All Files (*.*)\0*.*\0\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrDefExt = "jpg";
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY;

    if (!GetSaveFileNameA(&ofn))
    {
        return;
    }

    CudaArgs saveArgs = cpuArgs;
    saveArgs.scrwidth = saveWidth;
    saveArgs.scrheight = saveHeight;
    saveArgs.width = (xmax - xmin);
    saveArgs.height = (ymax - ymin);
    saveArgs.xmin = xmin;
    saveArgs.ymin = ymin;
    saveArgs.last = 1; // Force final-quality pass for saved image.
    saveArgs.iterations = getIterationsForZoom(getCurrentZoomFactor());

    const size_t imageSize = (size_t)saveWidth * (size_t)saveHeight * sizeof(unsigned int);
    std::vector<unsigned char> pixels(imageSize);
    cudaMandel(&saveArgs, pixels.data());

    const char* ext = strrchr(filename, '.');
    const bool saveAsBmp = ext && _stricmp(ext, ".bmp") == 0;
    bool saved = false;
    bool hasExif = false;

    if (saveAsBmp)
    {
        BITMAPFILEHEADER fileHeader = {};
        BITMAPINFOHEADER infoHeader = {};
        fileHeader.bfType = 0x4D42; // "BM"
        fileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
        fileHeader.bfSize = fileHeader.bfOffBits + (DWORD)imageSize;

        infoHeader.biSize = sizeof(BITMAPINFOHEADER);
        infoHeader.biWidth = saveWidth;
        infoHeader.biHeight = -saveHeight; // top-down rows
        infoHeader.biPlanes = 1;
        infoHeader.biBitCount = 32;
        infoHeader.biCompression = BI_RGB;
        infoHeader.biSizeImage = (DWORD)imageSize;

        FILE* f = nullptr;
        fopen_s(&f, filename, "wb");
        if (f)
        {
            fwrite(&fileHeader, sizeof(fileHeader), 1, f);
            fwrite(&infoHeader, sizeof(infoHeader), 1, f);
            fwrite(pixels.data(), imageSize, 1, f);
            fclose(f);
            saved = true;
        }
    }
    else
    {
        GdiplusStartupInput startupInput;
        ULONG_PTR gdiplusToken = 0;
        if (GdiplusStartup(&gdiplusToken, &startupInput, nullptr) == Ok)
        {
            {
                Bitmap image(saveWidth, saveHeight, PixelFormat32bppARGB);
                Rect rect(0, 0, saveWidth, saveHeight);
                BitmapData data = {};
                if (image.LockBits(&rect, ImageLockModeWrite, PixelFormat32bppARGB, &data) == Ok)
                {
                    for (int y = 0; y < saveHeight; ++y)
                    {
                        unsigned int* dst = (unsigned int*)((BYTE*)data.Scan0 + y * data.Stride);
                        const unsigned int* src = ((const unsigned int*)pixels.data()) + (size_t)y * (size_t)saveWidth;
                        for (int x = 0; x < saveWidth; ++x)
                        {
                            dst[x] = 0xFF000000u | src[x];
                        }
                    }
                    image.UnlockBits(&data);

                    const double left = saveArgs.xmin;
                    const double top = saveArgs.ymin;
                    const double right = saveArgs.xmin + saveArgs.width;
                    const double bottom = saveArgs.ymin + saveArgs.height;
                    const double centerX = (left + right) * 0.5;
                    const double centerY = (top + bottom) * 0.5;
                    const char* antialiasState = saveArgs.aa ? "ON" : "OFF";
                    const char* smoothState = saveArgs.smoothColoring ? "ON" : "OFF";
                    const char* paletteState = saveArgs.useCustomPalette ? "Custom" : "Default";

                    char dateTime[32];
                    sprintf_s(
                        dateTime,
                        "%04d:%02d:%02d %02d:%02d:%02d",
                        localTm.tm_year + 1900,
                        localTm.tm_mon + 1,
                        localTm.tm_mday,
                        localTm.tm_hour,
                        localTm.tm_min,
                        localTm.tm_sec);

                    char description[512];
                    sprintf_s(
                        description,
                        "Mandelbrot top-left=(%.15f, %.15f), bottom-right=(%.15f, %.15f)",
                        left,
                        top,
                        right,
                        bottom);

                    char userComment[2048];
                    sprintf_s(
                        userComment,
                        "Mandelbrot render; TopLeft=(%.15f, %.15f); BottomRight=(%.15f, %.15f); Center=(%.15f, %.15f); SaveSize=%dx%d; Iterations=%d; Zoom=%.6fx; IterationMultiplier=%.1f; Antialias=%s; SmoothColoring=%s; Palette=%s; GPU=%s; CUDACores=%d; CoreClockMHz=%.0f; MemoryClockMHz=%.0f; MemoryBandwidthGBs=%.1f",
                        left,
                        top,
                        right,
                        bottom,
                        centerX,
                        centerY,
                        saveWidth,
                        saveHeight,
                        saveArgs.iterations,
                        getCurrentZoomFactor(),
                        iterationScale,
                        antialiasState,
                        smoothState,
                        paletteState,
                        gpuName,
                        gpuCudaCores,
                        gpuCoreClockMHz,
                        gpuMemoryClockMHz,
                        gpuMemoryBandwidthGBs);

                    setAsciiExifProperty(image, 0x0131, "MandelCUDA"); // Software
                    setAsciiExifProperty(image, 0x0132, dateTime);      // DateTime
                    setAsciiExifProperty(image, 0x010E, description);   // ImageDescription
                    hasExif = setUserCommentExifProperty(image, userComment);

                    CLSID encoderClsid = {};
                    if (getEncoderClsid(L"image/jpeg", &encoderClsid) >= 0)
                    {
                        ULONG quality = 95;
                        EncoderParameters encoderParams = {};
                        encoderParams.Count = 1;
                        encoderParams.Parameter[0].Guid = EncoderQuality;
                        encoderParams.Parameter[0].Type = EncoderParameterValueTypeLong;
                        encoderParams.Parameter[0].NumberOfValues = 1;
                        encoderParams.Parameter[0].Value = &quality;

                        const std::wstring filenameW = toWideString(filename);
                        if (!filenameW.empty())
                        {
                            saved = (image.Save(filenameW.c_str(), &encoderClsid, &encoderParams) == Ok);
                        }
                    }
                }
            }
            GdiplusShutdown(gdiplusToken);
        }
    }

    if (saved)
    {
        char msg[768];
        if (saveAsBmp)
        {
            sprintf_s(msg, "Saved image:\n%s\n\nNote: BMP does not support EXIF metadata.", filename);
        }
        else
        {
            sprintf_s(
                msg,
                "Saved image:\n%s\n\nEXIF included: %s\n- Top-left corner\n- Bottom-right corner\n- Center / zoom / iterations\n- Palette and render options\n- GPU information",
                filename,
                hasExif ? "yes" : "partially");
        }
        MessageBoxA(hwnd, msg, "Save Complete", MB_OK | MB_ICONINFORMATION);
    }
    else
    {
        MessageBoxA(hwnd, "Failed to save output file.", "Save Failed", MB_OK | MB_ICONERROR);
    }
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

                double c = (double)xPos / renderWidth;
                xmin = xmin * (1.0 - c) + xmax * c - (c * newWidth);
                xmax = xmin + newWidth;

                c = (double)yPos / renderHeight;
                ymin = ymin * (1.0 - c) + ymax * c - (c * newHeight);
                ymax = ymin + newHeight;
            }
            else if (panMoved)
            {
                const int dx = xPos - panLastX;
                const int dy = yPos - panLastY;
                const double width = xmax - xmin;
                const double height = ymax - ymin;

                const double moveX = -(double)dx * width / renderWidth;
                const double moveY = -(double)dy * height / renderHeight;

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
        else if (wParam == 0x53)
        {
            //cpuArgs.smoothColoring = cpuArgs.smoothColoring ? 0 : 1;
            cpuArgs.smoothColoring = cpuArgs.smoothColoring ? 0 : 1;
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
        else if (wParam == VK_F2)
        {
            saveImage(hwnd);
        }
        else if (wParam == 0x52)
        {
            const bool shiftHeld = (GetKeyState(VK_SHIFT) & 0x8000) != 0;
            const int delta = shiftHeld ? -kRenderSizeStep : kRenderSizeStep;
            int newSize = renderWidth + delta;
            if (newSize < kRenderSizeMin)
            {
                newSize = kRenderSizeMin;
            }
            else if (newSize > kRenderSizeMax)
            {
                newSize = kRenderSizeMax;
            }
            setRenderWidth(hwnd, newSize);
        }
        else if (wParam == 0x4F)
        {
            xmin = kInitialXMin;
            xmax = kInitialXMax;
            ymin = kInitialYMin;
            ymax = kInitialYMax;
            syncViewAspectToRenderSize();

            zoom = 0;
            if (panning)
            {
                panning = false;
                ReleaseCapture();
            }
            xPos = renderWidth / 2;
            yPos = renderHeight / 2;
            panLastX = xPos;
            panLastY = yPos;

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

    cpuArgs.scrheight = renderHeight;
    cpuArgs.scrwidth = renderWidth;
    cpuArgs.aa = 1;
    cpuArgs.smoothColoring = 1;
    cpuArgs.useCustomPalette = 0;
    cpuArgs.iterations = getIterationsForZoom(getCurrentZoomFactor());
    xPos = renderWidth / 2;
    yPos = renderHeight / 2;
    panLastX = xPos;
    panLastY = yPos;
    syncViewAspectToRenderSize();

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int cores = getSPcores(prop);
    int coreClockKHz = 0;
    int memoryClockKHz = 0;
    int memoryBusWidthBits = 0;
    cudaDeviceGetAttribute(&coreClockKHz, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&memoryClockKHz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&memoryBusWidthBits, cudaDevAttrGlobalMemoryBusWidth, 0);
    strncpy_s(gpuName, prop.name, _TRUNCATE);
    gpuCudaCores = cores;
    gpuCoreClockMHz = (double)coreClockKHz / 1000.0;
    gpuMemoryClockMHz = (double)memoryClockKHz / 1000.0;
    gpuMemoryBandwidthGBs =
        ((double)memoryClockKHz * 1000.0 * ((double)memoryBusWidthBits / 8.0) * 2.0) / 1.0e9;

    char debug[256];
    sprintf(debug, "Cores: %d (major %d) (minor %d)\n", cores, prop.major, prop.minor);
    OutputDebugString(debug);


    // Create bitmap info
    ZeroMemory(&bi, sizeof(BITMAPINFO));
    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = renderWidth;
    bi.bmiHeader.biHeight = -renderHeight;  //negative so (0,0) is at top left
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
    RECT wr = { 0, 0, renderWidth, renderHeight };
    AdjustWindowRectEx(&wr, WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME, FALSE, WS_EX_CLIENTEDGE);
    const int windowWidth = wr.right - wr.left;
    const int windowHeight = wr.bottom - wr.top;
    const int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    const int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    const int windowX = (screenWidth - windowWidth) / 2;
    const int windowY = (screenHeight - windowHeight) / 2;

    HWND hwnd = CreateWindowEx(
        WS_EX_CLIENTEDGE,
        g_szClassName,
        "MandelCUDA v1.01",
        WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME,
        windowX, windowY, windowWidth, windowHeight,
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
    bi.bmiHeader.biWidth = renderWidth;
    bi.bmiHeader.biHeight = -renderHeight;  //negative so (0,0) is at top left
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;

    bitmap = ::CreateDIBSection(hDCMem, &bi, DIB_RGB_COLORS, (VOID**)&lpBitmapBits, NULL, 0);
    frameBitmap = CreateCompatibleBitmap(hdc, renderWidth, renderHeight);

    //memset(lpBitmapBits, 0, renderWidth * renderHeight * sizeof(long));

    //DeleteDC(hDCMem);
    ReleaseDC(hwnd, hdc);
}

void setRenderWidth(HWND hwnd, int newWidth)
{
    if (newWidth == renderWidth)
    {
        return;
    }

    renderWidth = newWidth;
    renderHeight = (renderWidth * 3) / 4;
    cpuArgs.scrwidth = renderWidth;
    cpuArgs.scrheight = renderHeight;
    xPos = renderWidth / 2;
    yPos = renderHeight / 2;
    panLastX = xPos;
    panLastY = yPos;
    syncViewAspectToRenderSize();

    initDIBS(hwnd);

    RECT wr = { 0, 0, renderWidth, renderHeight };
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


