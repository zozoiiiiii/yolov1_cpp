/************************************************************************/
/*
@author:  junliang
@brief:   
@time:    2019/03/20
*/
/************************************************************************/
#pragma once


#include <iostream>



class ImageUtil
{
public:
    struct ImageData
    {
        int h;
        int w;
        int c;
        float *data;
    };

    static void rgbgr_image(ImageData im);
    static ImageData make_empty_image(int w, int h, int c);
    static void free_image(ImageData m);
    static void draw_box(ImageData a, int x1, int y1, int x2, int y2, float r, float g, float b);
    static void draw_box_width(ImageData a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
    static ImageData make_image(int w, int h, int c);
    static float get_pixel(ImageData m, int x, int y, int c);
    static void set_pixel(ImageData m, int x, int y, int c, float val);
    static ImageData resize_image(ImageData im, int w, int h);
    static ImageData load_image(char *filename, int w, int h, int c);
    static ImageData load_image_stb(char *filename, int channels);
    static void save_image_png(ImageData im, const char *name);
    static void show_image(ImageData p, const char *name);

    static float get_color(int c, int x, int max);

    static void add_pixel(ImageData m, int x, int y, int c, float val);
    static ImageData copy_image(ImageData p);
    static void constrain_image(ImageData im);

    static ImageData get_label_v3(ImageData **characters, char *string, int size);
    static ImageData tile_images(ImageData a, ImageData b, int dx);
    static ImageData border_image(ImageData a, int border);
    static float get_pixel_extend(ImageUtil::ImageData m, int x, int y, int c);
    static void fill_cpu(int N, float ALPHA, float *X, int INCX);
    static void embed_image(ImageData source, ImageData dest, int dx, int dy);
    static void composite_image(ImageData source, ImageData dest, int dx, int dy);
    static void draw_label(ImageData a, int r, int c, ImageData label, const float *rgb);
    static ImageData **load_alphabet();
    static ImageUtil::ImageData load_image_color(char *filename, int w, int h);
};
