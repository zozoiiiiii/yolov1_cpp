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


 
class RouteLayer : public ILayer
{
public:
    virtual bool load(const IniParser* pParser, int section, size_params params);
    virtual void forward_layer_cpu(JJ::network* pNet, float *input, int train);

private:
    LayerData make_route_layer(int batch, int n, std::vector<int> input_layers, std::vector<int> input_sizes);
private:
    std::vector<int> m_input_layers; // route
    std::vector<int> m_input_sizes; // route 
};

NS_JJ_END