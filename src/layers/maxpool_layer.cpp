#include "maxpool_layer.h"

NS_JJ_BEGIN
LayerData MaxpoolLayer::make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    LayerData l = { 0 };
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size) / stride + 1;
    l.out_h = (h + padding - size) / stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h * w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    m_indexes = (int*)calloc(output_size, sizeof(int));
    l.output = (float*)calloc(output_size, sizeof(float));
    //l.output_int8.assign(output_size, 0);
    //l.delta = calloc(output_size, sizeof(float));
    // commented only for this custom version of Yolo v2
    //l.forward = forward_maxpool_layer;
    //l.backward = backward_maxpool_layer;
    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

bool MaxpoolLayer::load(const IniParser* pParser, int section, size_params params)
{
    int stride = pParser->ReadInteger(section, "stride", 1);
    int size = pParser->ReadInteger(section, "size", stride);
    int padding = pParser->ReadInteger(section, "padding", size - 1);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c))
        return false;// error("Layer before maxpool LayerData must output image.");

    m_ld = make_maxpool_layer(batch, h, w, c, size, stride, padding);
    setType(MAXPOOL);
    return true;
}



void forward_maxpool_layer_avx(float *src, float *dst, int* indexes, int size, int w, int h, int out_w, int out_h, int c,
    int pad, int stride, int batch)
{
    int b, k;
    const int w_offset = -pad / 2;
    const int h_offset = -pad / 2;

    for (b = 0; b < batch; ++b)
    {
        for (k = 0; k < c; ++k)
        {
            int i, j, m, n;
            for (i = 0; i < out_h; ++i)
            {
                for (j = 0; j < out_w; ++j)
                {
                    int out_index = j + out_w * (i + out_h * (k + c * b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for (n = 0; n < size; ++n)
                    {
                        for (m = 0; m < size; ++m) {
                            int cur_h = h_offset + i * stride + n;
                            int cur_w = w_offset + j * stride + m;
                            int index = cur_w + w * (cur_h + h * (k + b * c));
                            int valid = (cur_h >= 0 && cur_h < h &&
                                cur_w >= 0 && cur_w < w);
                            float val = (valid != 0) ? src[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max = (val > max) ? val : max;
                        }
                    }
                    dst[out_index] = max;
                    indexes[out_index] = max_i;
                }
            }
        }
    }
}

void MaxpoolLayer::forward_layer_cpu(JJ::network* pNet, float *input, int train)
{
    LayerData& l = m_ld;

    if (!train)
    {
        forward_maxpool_layer_avx(input, l.output, m_indexes, l.size, l.w, l.h, l.out_w, l.out_h, l.c, l.pad, l.stride, l.batch);
        return;
    }

    int b, i, j, k, m, n;
    const int w_offset = -l.pad;
    const int h_offset = -l.pad;

    const int h = l.out_h;
    const int w = l.out_w;
    const int c = l.c;

    // batch index
    for (b = 0; b < l.batch; ++b) {
        // channel index
        for (k = 0; k < c; ++k) {
            // y - input
            for (i = 0; i < h; ++i) {
                // x - input
                for (j = 0; j < w; ++j) {
                    int out_index = j + w * (i + h * (k + c * b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    // pooling x-index
                    for (n = 0; n < l.size; ++n) {
                        // pooling y-index
                        for (m = 0; m < l.size; ++m) {
                            int cur_h = h_offset + i * l.stride + n;
                            int cur_w = w_offset + j * l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b * l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;    // get max index
                            max = (val > max) ? val : max;            // get max value
                        }
                    }
                    l.output[out_index] = max;        // store max value
                    m_indexes[out_index] = max_i;    // store max index
                }
            }
        }
    }
}
NS_JJ_END