/************************************************************************/
/*
@author:  junliang
@brief:
@time:    2019/02/22
*/
/************************************************************************/
#pragma once

#include <string>
#include "layers/convolutional_layer.h"
#include "parser/ini_parser.h"
#include "utils/string_util.h"
#include "box.h"

#include "utils/image_util.h"

NS_JJ_BEGIN

struct detection_with_class
{
    detection det;
    // The most probable class id: the best class index in this->prob.
    // Is filled temporary when processing results, otherwise not initialized
    int best_class;
};

class Detector
{
public:
    static Detector* instance();
    bool load(char **names, char *cfgfile, char *weightfile) { return false; }

    // detect one image
    bool test(const char* datacfg, char *cfgfile, char *weightfile, char *filename, float thresh);

    // train some images, get one weight file.
    bool train(const char* datacfg, char *cfgfile, char *weightfile);

private:
    JJ::network* readConfigFile(const char* cfgFile, int batch);
    bool readWeightFile(network *net, char *filename, int cutoff);
    bool parseNetOptions(const IniParser* pIniParser, JJ::network* pNetWork);
    void draw_detections_v3(ImageUtil::ImageData im, detection *dets, int num, float thresh, const std::vector<std::string>& names, ImageUtil::ImageData **alphabet, int classes, int ext_output);
    

    JJ::LAYER_TYPE string_to_layer_type(const std::string& type);



    detection_with_class* get_actual_detections(detection *dets, int dets_num, float thresh, int* selected_detections_num);





    void yolov2_fuse_conv_batchnorm(network* net);

    void calculate_binary_weights(network* net);



    void network_predict_cpu(network* net, float *input);


    void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets, int letter);
    detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter);



private:
    JJ::network* m_pNet;
};
NS_JJ_END