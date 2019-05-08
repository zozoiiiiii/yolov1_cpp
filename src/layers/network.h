/************************************************************************/
/*
@author:  junliang
@brief:   
@time:    2019/02/22
*/
/************************************************************************/
#pragma once

#include <string>
#include <vector>
#include <map>
#include "layer.h"



NS_JJ_BEGIN



enum learning_rate_policy
{
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
};

struct network
{
    float* workspace;   // save the data and transfer between layers
    int n;  // LayerData count

    int batch;
    std::vector<ILayer*> jjLayers;
    uint64_t *seen;                 // weight file, not in use

    int outputs;
    float* output;
    learning_rate_policy policy;

    int time_steps;

    int adam;
    float B1;
    float B2;

    int inputs;
    int h, w, c;
};



class NetWork
{
public:
    static int get_network_output_size(network* pNet);
    static float* get_network_output(network* pNet);
};
NS_JJ_END