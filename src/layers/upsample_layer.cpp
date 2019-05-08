#include "upsample_layer.h"

NS_JJ_BEGIN

UpsampleLayer::UpsampleLayer():m_scale(0),m_reverse(0)
{}

LayerData UpsampleLayer::make_upsample_layer(int batch, int w, int h, int c, int stride)
{
    LayerData l = { 0 };
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w * stride;
    l.out_h = h * stride;
    l.out_c = c;
    if (stride < 0) {
        stride = -stride;
        m_reverse = 1;
        l.out_w = w / stride;
        l.out_h = h / stride;
    }
    l.stride = stride;
    l.outputs = l.out_w*l.out_h*l.out_c;
    l.inputs = l.w*l.h*l.c;
    //l.delta = calloc(l.outputs*batch, sizeof(float));
    l.output = (float*)calloc(l.outputs*batch, sizeof(float));

    //l.forward = forward_upsample_layer;
    if (m_reverse)
        fprintf(stderr, "downsample         %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    else
        fprintf(stderr, "upsample           %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

bool UpsampleLayer::load(const IniParser* pParser, int section, size_params params)
{
    int stride = pParser->ReadInteger(section, "stride", 2);
    m_ld = make_upsample_layer(params.batch, params.w, params.h, params.c, stride);
    setType(UPSAMPLE);
    m_scale = pParser->ReadFloat(section, "scale", 1);
    //return LayerData;
    return true;
}


void UpsampleLayer::resize_upsample_layer(LayerData *l, int w, int h)
{
  
}

void UpsampleLayer::forward_upsample_layer(const LayerData l, network net)
{
}

void UpsampleLayer::backward_upsample_layer(const LayerData l, network net)
{
   
}

void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for (i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    int i, j, k, b;
    for (b = 0; b < batch; ++b) {
        for (k = 0; k < c; ++k) {
            for (j = 0; j < h*stride; ++j) {
                for (i = 0; i < w*stride; ++i) {
                    int in_index = b * w*h*c + k * w*h + (j / stride)*w + i / stride;
                    int out_index = b * w*h*c*stride*stride + k * w*h*stride*stride + j * w*stride + i;
                    if (forward) out[out_index] = scale * in[in_index];
                    else in[in_index] += scale * out[out_index];
                }
            }
        }
    }
}

void UpsampleLayer::forward_layer_cpu(JJ::network* pNet, float *input, int train)
{
    LayerData& l = m_ld;
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    if (m_reverse)
    {
        upsample_cpu(l.output, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, m_scale, input);
    }
    else
    {
        upsample_cpu(input, l.w, l.h, l.c, l.batch, l.stride, 1, m_scale, l.output);
    }
}
NS_JJ_END