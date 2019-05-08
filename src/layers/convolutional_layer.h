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
#include <intrin.h> 

NS_JJ_BEGIN



enum ACTIVATION
{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
};
struct ConvolutionalLayerInfo
{
    char *align_bit_weights;
    float* mean_arr;    // conv
    int align_bit_weights_size;
    int lda_align;
    int bit_align;

    int use_bin_output;

    float bflops;

    ///////
    int filters;
    int size;
    int stride;
    int pad;
    int padding;
    ACTIVATION activation;

    int batch_normalize;
    int binary;
    int xnor;
    int bin_output;

    int flipped;
    float dot;

    ConvolutionalLayerInfo() :filters(1), size(1), stride(1), pad(0), padding(0), activation(LOGISTIC), batch_normalize(0),
        binary(0), xnor(0), bin_output(0), flipped(0), dot(0) {}
};


struct ConvolutionWeight
{
    float* biases;
    float* scales;
    float* rolling_mean;
    float* rolling_variance;
    float *weights;

    // tempory
    float* binary_weights;
    float *binary_input;
};






typedef LayerData convolutional_layer;
class ConvolutionLayer : public ILayer
{
public:
    static void activate_array(float *x, const int n, const ACTIVATION a);
public:
    virtual bool load(const IniParser* pParser, int section, size_params params);
    virtual void forward_layer_cpu(JJ::network* pNet, float *input, int train);
    ConvolutionWeight* getWeight() { return &m_weight; }
    ConvolutionalLayerInfo* getConv() { return &m_conv; }

    void load_convolutional_weights_cpu(FILE *fp);
    void binary_align_weights();
    void fuse_batchnorm();
private:
    void make_convolutional_layer();
    size_t ConvolutionLayer::get_workspace_size();
private:
    ConvolutionalLayerInfo m_conv;
    ConvolutionWeight m_weight;
};

NS_JJ_END