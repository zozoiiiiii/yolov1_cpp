/************************************************************************/
/*
@author:  junliang
@brief:   convolution LayerData
@time:    2019/02/22
*/
/************************************************************************/
#pragma once

#include <string>
#include <vector>
#include <map>
#include "layer.h"
#include "network.h"


NS_JJ_BEGIN



class UpsampleLayer : public ILayer
{
public:
    UpsampleLayer();
    virtual bool load(const IniParser* pParser, int section, size_params params);
    virtual void forward_layer_cpu(JJ::network* pNet, float *input, int train);


    void forward_upsample_layer(const LayerData l, network net);
    void backward_upsample_layer(const LayerData l, network net);
    void resize_upsample_layer(LayerData *l, int w, int h);
private:
    LayerData make_upsample_layer(int batch, int w, int h, int c, int stride);
private:
    float m_scale;    // unsample
    int m_reverse;
};

NS_JJ_END