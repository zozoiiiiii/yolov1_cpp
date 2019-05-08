#include "image_util.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>


#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"



void ImageUtil::rgbgr_image(ImageData im)
{
    int i;
    for (i = 0; i < im.w*im.h; ++i)
    {
        float swap = im.data[i];
        im.data[i] = im.data[i + im.w*im.h * 2];
        im.data[i + im.w*im.h * 2] = swap;
    }
}

ImageUtil::ImageData ImageUtil::make_empty_image(int w, int h, int c)
{
    ImageData out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

void ImageUtil::free_image(ImageData m)
{
    if (m.data) {
        free(m.data);
    }
}

// image.c
void ImageUtil::draw_box(ImageData a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    //normalize_image(a);
    int i;
    if (x1 < 0) x1 = 0;
    if (x1 >= a.w) x1 = a.w - 1;
    if (x2 < 0) x2 = 0;
    if (x2 >= a.w) x2 = a.w - 1;

    if (y1 < 0) y1 = 0;
    if (y1 >= a.h) y1 = a.h - 1;
    if (y2 < 0) y2 = 0;
    if (y2 >= a.h) y2 = a.h - 1;

    for (i = x1; i <= x2; ++i) {
        a.data[i + y1 * a.w + 0 * a.w*a.h] = r;
        a.data[i + y2 * a.w + 0 * a.w*a.h] = r;

        a.data[i + y1 * a.w + 1 * a.w*a.h] = g;
        a.data[i + y2 * a.w + 1 * a.w*a.h] = g;

        a.data[i + y1 * a.w + 2 * a.w*a.h] = b;
        a.data[i + y2 * a.w + 2 * a.w*a.h] = b;
    }
    for (i = y1; i <= y2; ++i) {
        a.data[x1 + i * a.w + 0 * a.w*a.h] = r;
        a.data[x2 + i * a.w + 0 * a.w*a.h] = r;

        a.data[x1 + i * a.w + 1 * a.w*a.h] = g;
        a.data[x2 + i * a.w + 1 * a.w*a.h] = g;

        a.data[x1 + i * a.w + 2 * a.w*a.h] = b;
        a.data[x2 + i * a.w + 2 * a.w*a.h] = b;
    }
}

// image.c
void ImageUtil::draw_box_width(ImageData a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
    int i;
    for (i = 0; i < w; ++i) {
        draw_box(a, x1 + i, y1 + i, x2 - i, y2 - i, r, g, b);
    }
}

float ImageUtil::get_pixel_extend(ImageUtil::ImageData m, int x, int y, int c)
{
    if (x < 0 || x >= m.w || y < 0 || y >= m.h) return 0;
    /*
    if(x < 0) x = 0;
    if(x >= m.w) x = m.w-1;
    if(y < 0) y = 0;
    if(y >= m.h) y = m.h-1;
    */
    if (c < 0 || c >= m.c) return 0;
    return get_pixel(m, x, y, c);
}

ImageUtil::ImageData ImageUtil::border_image(ImageData a, int border)
{
    ImageData b = make_image(a.w + 2 * border, a.h + 2 * border, a.c);
    int x, y, k;
    for (k = 0; k < b.c; ++k)
    {
        for (y = 0; y < b.h; ++y)
        {
            for (x = 0; x < b.w; ++x)
            {
                float val = get_pixel_extend(a, x - border, y - border, k);
                if (x - border < 0 || x - border >= a.w || y - border < 0 || y - border >= a.h) val = 1;
                set_pixel(b, x, y, k, val);
            }
        }
    }
    return b;
}

void ImageUtil::fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for (i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}


void ImageUtil::embed_image(ImageData source, ImageData dest, int dx, int dy)
{
    int x, y, k;
    for (k = 0; k < source.c; ++k) {
        for (y = 0; y < source.h; ++y) {
            for (x = 0; x < source.w; ++x) {
                float val = get_pixel(source, x, y, k);
                set_pixel(dest, dx + x, dy + y, k, val);
            }
        }
    }
}


void ImageUtil::composite_image(ImageData source, ImageData dest, int dx, int dy)
{
    int x, y, k;
    for (k = 0; k < source.c; ++k) {
        for (y = 0; y < source.h; ++y) {
            for (x = 0; x < source.w; ++x) {
                float val = get_pixel(source, x, y, k);
                float val2 = get_pixel_extend(dest, dx + x, dy + y, k);
                set_pixel(dest, dx + x, dy + y, k, val * val2);
            }
        }
    }
}


ImageUtil::ImageData ImageUtil::tile_images(ImageData a, ImageData b, int dx)
{
    if (a.w == 0)
        return copy_image(b);

    ImageData c = make_image(a.w + b.w + dx, (a.h > b.h) ? a.h : b.h, (a.c > b.c) ? a.c : b.c);
    fill_cpu(c.w*c.h*c.c, 1, c.data, 1);
    embed_image(a, c, 0, 0);
    composite_image(b, c, a.w + dx, 0);
    return c;
}


ImageUtil::ImageData ImageUtil::get_label_v3(ImageData **characters, char *string, int size)
{
    size = size / 10;
    if (size > 7) size = 7;
    ImageData label = make_empty_image(0, 0, 0);
    while (*string)
    {
        ImageData l = characters[size][(int)*string];
        ImageData n = tile_images(label, l, -size - 1 + (size + 1) / 2);
        free_image(label);
        label = n;
        ++string;
    }
    ImageData b = border_image(label, label.h*.25);
    free_image(label);
    return b;
}


void ImageUtil::draw_label(ImageData a, int r, int c, ImageData label, const float *rgb)
{
    int w = label.w;
    int h = label.h;
    if (r - h >= 0) r = r - h;

    int i, j, k;
    for (j = 0; j < h && j + r < a.h; ++j) {
        for (i = 0; i < w && i + c < a.w; ++i) {
            for (k = 0; k < label.c; ++k) {
                float val = get_pixel(label, i, j, k);
                set_pixel(a, i + c, j + r, k, rgb[k] * val);
            }
        }
    }
}

// image.c
ImageUtil::ImageData ImageUtil::make_image(int w, int h, int c)
{
    ImageData out = make_empty_image(w, h, c);
    out.data = (float*)calloc(h*w*c, sizeof(float));
    return out;
}

// image.c
float ImageUtil::get_pixel(ImageData m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y * m.w + x];
}

// image.c
void ImageUtil::set_pixel(ImageData m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y * m.w + x] = val;
}

// image.c
void ImageUtil::add_pixel(ImageData m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y * m.w + x] += val;
}

// image.c
ImageUtil::ImageData ImageUtil::resize_image(ImageData im, int w, int h)
{
    ImageData resized = make_image(w, h, im.c);
    ImageData part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < im.h; ++r) {
            for (c = 0; c < w; ++c) {
                float val = 0;
                if (c == w - 1 || im.w == 1) {
                    val = get_pixel(im, im.w - 1, r, k);
                }
                else {
                    float sx = c * w_scale;
                    int ix = (int)sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix + 1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < h; ++r) {
            float sy = r * h_scale;
            int iy = (int)sy;
            float dy = sy - iy;
            for (c = 0; c < w; ++c) {
                float val = (1 - dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if (r == h - 1 || im.h == 1) continue;
            for (c = 0; c < w; ++c) {
                float val = dy * get_pixel(part, c, iy + 1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}

// image.c
ImageUtil::ImageData ImageUtil::load_image(char *filename, int w, int h, int c)
{
#ifdef OPENCV
    ImageData out = load_image_cv(filename, c);
#else
    ImageData out = load_image_stb(filename, c);
#endif

    if ((h && w) && (h != out.h || w != out.w)) {
        ImageData resized = resize_image(out, w, h);
        free_image(out);
        out = resized;
    }
    return out;
}

// image.c
ImageUtil::ImageData ImageUtil::load_image_stb(char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load ImageData \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }
    if (channels) c = channels;
    int i, j, k;
    ImageData im = make_image(w, h, c);
    for (k = 0; k < c; ++k) {
        for (j = 0; j < h; ++j) {
            for (i = 0; i < w; ++i) {
                int dst_index = i + w * j + w * h*k;
                int src_index = k + c * i + c * w*j;
                im.data[dst_index] = (float)data[src_index] / 255.;
            }
        }
    }
    free(data);
    return im;
}


// image.c
ImageUtil::ImageData ImageUtil::copy_image(ImageData p)
{
    ImageUtil::ImageData copy = p;
    copy.data = (float*)calloc(p.h*p.w*p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h*p.w*p.c * sizeof(float));
    return copy;
}

// image.c
void ImageUtil::constrain_image(ImageData im)
{
    int i;
    for (i = 0; i < im.w*im.h*im.c; ++i) {
        if (im.data[i] < 0) im.data[i] = 0;
        if (im.data[i] > 1) im.data[i] = 1;
    }
}

// image.c
void ImageUtil::save_image_png(ImageData im, const char *name)
{
    char buff[256];
    sprintf(buff, "%s.png", name);
    unsigned char *data = (unsigned char*)calloc(im.w*im.h*im.c, sizeof(char));
    int i, k;
    for (k = 0; k < im.c; ++k) {
        for (i = 0; i < im.w*im.h; ++i) {
            data[i*im.c + k] = (unsigned char)(255 * im.data[i + k * im.w*im.h]);
        }
    }
    int success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
    free(data);
    if (!success) fprintf(stderr, "Failed to write ImageData %s\n", buff);
}


// image.c
void ImageUtil::show_image(ImageData p, const char *name)
{
#ifdef OPENCV
    show_image_cv(p, name);
#else
    fprintf(stderr, "Not compiled with OpenCV, saving to %s.png instead\n", name);
    save_image_png(p, name);
#endif
}

// image.c
float ImageUtil::get_color(int c, int x, int max)
{
    static float colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
    float ratio = ((float)x / max) * 5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
    //printf("%f\n", r);
    return r;
}



ImageUtil::ImageData ** ImageUtil::load_alphabet()
{
    int i, j;
    const int nsize = 8;
    ImageData** alphabets = (ImageData**)calloc(nsize, sizeof(ImageData*));
    for (j = 0; j < nsize; ++j) {
        alphabets[j] = (ImageData*)calloc(128, sizeof(ImageData));
        for (i = 32; i < 127; ++i) {
            char buff[256];
            sprintf(buff, "data/labels/%d_%d.png", i, j);
            alphabets[j][i] = load_image_color(buff, 0, 0);
        }
    }
    return alphabets;
}

ImageUtil::ImageData ImageUtil::load_image_color(char *filename, int w, int h)
{
    return load_image(filename, w, h, 3);
}