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
#include "../parser/ini_parser.h"
#include "../utils/string_util.h"

// namespace
#ifndef NS_JJ_BEGIN
#define NS_JJ_BEGIN                     namespace JJ {
#define NS_JJ_END                       }
#define USING_NS_JJ                     using namespace JJ;
#endif


NS_JJ_BEGIN

class network;


typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    UPSAMPLE,
    REORG,
    BLANK
} LAYER_TYPE;





struct LayerData
{
    int batch;
    int inputs;
    int outputs;

    // channels of input-array
    // height of input-array
    // width of input-array
    int h, w, c;
    int out_h, out_w, out_c;
    int n;          // filter size
    int size;       // width and height of filters (the same size for all filters)
    int stride;
    int pad;

    int dontload;   // this conv LayerData not need to load weight information
    int dontloadscales;

    float* output;  // con, yolo, maxpool, route, upsample
    size_t workspace_size;
};




struct size_params
{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    JJ::network* net;
};

class ILayer
{
public:
    virtual bool load(const IniParser* pParser, int section, size_params params) = 0;

    // used for detect
    virtual void forward_layer_cpu(JJ::network* pNet, float *input, int train) {}

    // used for train
    //virtual void backward_layer_cpu(JJ::network* pNet, float *input, int train) {}
    //virtual void update_layer_cpu(JJ::network* pNet, float *input, int train) {}


    LayerData* getLayer() { return &m_ld; }

    LAYER_TYPE getType() { return m_type; }
    void setType(LAYER_TYPE lt) { m_type = lt; }

protected:
    LayerData m_ld;
    LAYER_TYPE m_type;
};

NS_JJ_END