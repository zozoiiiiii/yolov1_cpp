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
#include "../box.h"

NS_JJ_BEGIN



typedef LayerData convolutional_layer;
class YoloLayer : public ILayer
{
public:
    virtual bool load(const IniParser* pParser, int section, size_params params);
    virtual void forward_layer_cpu(JJ::network* pNet, float *input, int train);

    int get_yolo_detections(int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter);
    int getClasses() { return m_classes; }
    int entry_index(int batch, int location, int entry);
private:
    LayerData make_yolo_layer(int batch, int w, int h, int n, int total, std::vector<int> mask, int classes, int max_boxes);
private:
    int m_classes;
    std::vector<int> m_mask;
    std::vector<float> m_biases;
};

NS_JJ_END