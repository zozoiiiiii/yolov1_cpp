#include "detector.h"
#include "parser/ini_parser.h"
#include "layers/convolutional_layer.h"
#include "layers/yolo_layer.h"
#include "layers/maxpool_layer.h"
#include "layers/route_layer.h"
#include "layers/upsample_layer.h"
#include <time.h>

NS_JJ_BEGIN






// global GPU index: cuda.c
int gpu_index = 0;

// Creates array of detections with prob > thresh and fills best_class for them
detection_with_class* Detector::get_actual_detections(detection *dets, int dets_num, float thresh, int* selected_detections_num)
{
    int selected_num = 0;
    detection_with_class* result_arr = (detection_with_class*)calloc(dets_num, sizeof(detection_with_class));
    int i;
    for (i = 0; i < dets_num; ++i)
    {
        int best_class = -1;
        float best_class_prob = thresh;
        int j;
        for (j = 0; j < dets[i].classes; ++j)
        {
            if (dets[i].prob[j] > best_class_prob) {
                best_class = j;
                best_class_prob = dets[i].prob[j];
            }
        }

        if (best_class >= 0)
        {
            result_arr[selected_num].det = dets[i];
            result_arr[selected_num].best_class = best_class;
            ++selected_num;
        }
    }
    if (selected_detections_num)
        *selected_detections_num = selected_num;
    return result_arr;
}

// compare to sort detection** by bbox.x
int compare_by_lefts(const void *a_ptr, const void *b_ptr)
{
    const detection_with_class* a = (detection_with_class*)a_ptr;
    const detection_with_class* b = (detection_with_class*)b_ptr;
    const float delta = (a->det.bbox.x - a->det.bbox.w / 2) - (b->det.bbox.x - b->det.bbox.w / 2);
    return delta < 0 ? -1 : delta > 0 ? 1 : 0;
}

// compare to sort detection** by best_class probability
int compare_by_probs(const void *a_ptr, const void *b_ptr)
{
    const detection_with_class* a = (detection_with_class*)a_ptr;
    const detection_with_class* b = (detection_with_class*)b_ptr;
    float delta = a->det.prob[a->best_class] - b->det.prob[b->best_class];
    return delta < 0 ? -1 : delta > 0 ? 1 : 0;
}

void Detector::draw_detections_v3(ImageUtil::ImageData im, detection *dets, int num, float thresh, const std::vector<std::string>& names,
    ImageUtil::ImageData **alphabet, int classes, int ext_output)
{
    int selected_detections_num;
    detection_with_class* selected_detections = get_actual_detections(dets, num, thresh, &selected_detections_num);

    // text output
    qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_lefts);
    int i;
    for (i = 0; i < selected_detections_num; ++i)
    {
        const int best_class = selected_detections[i].best_class;
        printf("%s: %.0f%%", names[best_class], selected_detections[i].det.prob[best_class] * 100);
        if (ext_output)
            printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
                round((selected_detections[i].det.bbox.x - selected_detections[i].det.bbox.w / 2)*im.w),
                round((selected_detections[i].det.bbox.y - selected_detections[i].det.bbox.h / 2)*im.h),
                round(selected_detections[i].det.bbox.w*im.w), round(selected_detections[i].det.bbox.h*im.h));
        else
            printf("\n");
        int j;
        for (j = 0; j < classes; ++j) {
            if (selected_detections[i].det.prob[j] > thresh && j != best_class) {
                printf("%s: %.0f%%\n", names[j], selected_detections[i].det.prob[j] * 100);
            }
        }
    }

    // ImageData output
    qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_probs);
    for (i = 0; i < selected_detections_num; ++i)
    {
        int width = im.h * .006;
        if (width < 1)
            width = 1;

        /*
        if(0){
        width = pow(prob, 1./2.)*10+1;
        alphabet = 0;
        }
        */

        //printf("%d %s: %.0f%%\n", i, names[selected_detections[i].best_class], prob*100);
        int offset = selected_detections[i].best_class * 123457 % classes;
        float red = ImageUtil::get_color(2, offset, classes);
        float green = ImageUtil::get_color(1, offset, classes);
        float blue = ImageUtil::get_color(0, offset, classes);
        float rgb[3];

        //width = prob*20+2;

        rgb[0] = red;
        rgb[1] = green;
        rgb[2] = blue;
        box b = selected_detections[i].det.bbox;
        //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

        int left = (b.x - b.w / 2.)*im.w;
        int right = (b.x + b.w / 2.)*im.w;
        int top = (b.y - b.h / 2.)*im.h;
        int bot = (b.y + b.h / 2.)*im.h;

        if (left < 0) left = 0;
        if (right > im.w - 1) right = im.w - 1;
        if (top < 0) top = 0;
        if (bot > im.h - 1) bot = im.h - 1;

        ImageUtil::draw_box_width(im, left, top, right, bot, width, red, green, blue);


        if (alphabet)
        {
            char labelstr[4096] = { 0 };
            strcat(labelstr, names[selected_detections[i].best_class].c_str());
            int j;
            for (j = 0; j < classes; ++j) {
                if (selected_detections[i].det.prob[j] > thresh && j != selected_detections[i].best_class) {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j].c_str());
                }
            }
            
            ImageUtil::ImageData label = ImageUtil::get_label_v3(alphabet, labelstr, (im.h*.03));
            ImageUtil::draw_label(im, top + width, left, label, rgb);
            ImageUtil::free_image(label);
        }
    }
    free(selected_detections);
}


// fuse convolutional and batch_norm weights into one convolutional-LayerData
void Detector::yolov2_fuse_conv_batchnorm(network* net)
{
    int j;

    // visit all layers
    for (j = 0; j < net->n; ++j)
    {
        ILayer* pLayer = net->jjLayers[j];
        LayerData *l = pLayer->getLayer();
        if (pLayer->getType() == CONVOLUTIONAL)
        {
            printf(" Fuse Convolutional LayerData \t\t l->size = %d  \n", l->size);
            ConvolutionLayer* pConv = (ConvolutionLayer*)pLayer;
            pConv->fuse_batchnorm();
        }
        else {
            printf(" Skip LayerData: %d \n", pLayer->getType());
        }
    }  
}

void Detector::calculate_binary_weights(network* net)
{
    int j;
    for (j = 0; j < net->n; ++j)
    {
        ILayer* pLayer = net->jjLayers[j];
        LayerData *l = pLayer->getLayer();
        if (pLayer->getType() == CONVOLUTIONAL)
        {
            //printf(" Merges Convolutional-%d and batch_norm \n", j);
            ConvolutionLayer* pConv = (ConvolutionLayer*)pLayer;
            if (pConv->getConv()->xnor)
            {
                //printf("\n %d \n", j);
                pConv->getConv()->lda_align = 256; // 256bit for AVX2

                ConvolutionLayer* pConvLayer = (ConvolutionLayer*)pLayer;
                pConvLayer->binary_align_weights();

                if (pConv->getConv()->use_bin_output)
                {
                    pConv->getConv()->activation = LINEAR;
                }
            }
        }
    }
}


Detector* Detector::instance()
{
    static Detector s_detector;
    return &s_detector;
}

bool Detector::test(const char* datacfg, char *cfgfile, char *weightfile, char *filename, float thresh)
{
    // load object names
    std::vector<std::string> names;
    int obj_count = 0;
    FILE* fp;
    char buffer[255];
    fp = fopen(datacfg, "r");
    while (fgets(buffer, 255, (FILE*)fp))
    {
        buffer[strlen(buffer) - 1] = '\0';  // remove newline
        names.push_back(buffer);
    }
    fclose(fp);

    if (!filename)
        return false;

    ImageUtil::ImageData **alphabet = ImageUtil::load_alphabet();

    // 1. read config file, like convolution LayerData
    network* net = readConfigFile(cfgfile, 1);    // parser.c
    if (weightfile)
    {
        // 2. read weight file, init the LayerData information in the network. cutoff == net.n, means do not cut off any LayerData
        readWeightFile(net, weightfile, net->n);
    }




    //set_batch_network(&net, 1);                    // network.c
    srand(2222222);
    yolov2_fuse_conv_batchnorm(net);
    calculate_binary_weights(net);


    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms = .4;

        // 3. open image
        strncpy(input, filename, 256);
        ImageUtil::ImageData im = ImageUtil::load_image(input, 0, 0, 3);            // image.c
        ImageUtil::ImageData sized = ImageUtil::resize_image(im, net->w, net->h);    // image.c


        float *X = sized.data;
        time = clock();

        // 4. predict, the key
        network_predict_cpu(net, X);

        printf("%s: Predicted in %f seconds.\n", input, (float)(clock() - time) / CLOCKS_PER_SEC); //sec(clock() - time));

        // 5. save to ImageData or show directly
        float hier_thresh = 0.5;
        int ext_output = 1, letterbox = 0, nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);

        ILayer* pLayer = net->jjLayers[net->n - 1];
        if (pLayer->getType() != YOLO)
            return false;

        YoloLayer* pYolo = (YoloLayer*)pLayer;
        if (nms)
            do_nms_sort(dets, nboxes, pYolo->getClasses(), nms);

        draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, pYolo->getClasses(), ext_output);
        ImageUtil::save_image_png(im, "predictions");    // image.c
        ImageUtil::free_image(im);                    // image.c
        ImageUtil::free_image(sized);                // image.c
        return true;

}




bool Detector::train(const char* datacfg, char *cfgfile, char *weightfile)
{

    return true;

}

void Detector::fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets, int letter)
{
    int j;
    for (j = 0; j < net->n; ++j)
    {
        ILayer* pLayer = net->jjLayers[j];
        LayerData* l = pLayer->getLayer();
        if (pLayer->getType() == YOLO)
        {
            YoloLayer* pYoloLayer = (YoloLayer*)pLayer;
            int count = pYoloLayer->get_yolo_detections(w, h, net->w, net->h, thresh, map, relative, dets, letter);
            dets += count;
        }
    }
}

int yolo_num_detections(YoloLayer* pLayer, float thresh)
{
    LayerData* pLayerInfo = pLayer->getLayer();

    int i, n;
    int count = 0;
    for (i = 0; i < pLayerInfo->w* pLayerInfo->h; ++i)
    {
        for (n = 0; n < pLayerInfo->n; ++n)
        {
            int obj_index =  pLayer->entry_index(0, n* pLayerInfo->w* pLayerInfo->h + i, 4);

            // detect something
            float val = pLayerInfo->output[obj_index];
            if (val > thresh)
            {
                ++count;
            }
        }
    }
    return count;
}

int num_detections(network *net, float thresh)
{
    int i;
    int s = 0;
    for (i = 0; i < net->n; ++i)
    {
        ILayer* pLayer = net->jjLayers[i];
        LayerData* pLayerInfo = pLayer->getLayer();
        if (pLayer->getType() == YOLO)
        {
            YoloLayer* pYolo = (YoloLayer*)pLayer;
            s += yolo_num_detections(pYolo, thresh);
        }

        if (pLayer->getType() == DETECTION || pLayer->getType() == REGION) {
            s += pLayerInfo->w * pLayerInfo->h * pLayerInfo->n;
        }
    }
    return s;
}

detection *make_network_boxes(network *net, float thresh, int *num)
{
    ILayer* pLayer = net->jjLayers[net->n - 1]; // the last LayerData always yolo
    if (pLayer->getType() != YOLO)
        return nullptr;

    YoloLayer* pYolo = (YoloLayer*)pLayer;
    LayerData* pLayerInfo = pLayer->getLayer();
    int i;
    int nboxes = num_detections(net, thresh);
    if (num) *num = nboxes;
    detection *dets = (detection*)calloc(nboxes, sizeof(detection));
    for (i = 0; i < nboxes; ++i)
    {
        dets[i].prob = (float*)calloc(pYolo->getClasses(), sizeof(float));
    }
    return dets;
}

detection * Detector::get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter)
{
    detection *dets = make_network_boxes(net, thresh, num);
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets, letter);
    return dets;
}



bool Detector::readWeightFile(network *net, char *filename, int cutoff)
{
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if (!fp)
        return false;

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major * 10 + minor) >= 2)
    {
        fread(net->seen, sizeof(uint64_t), 1, fp);
    }
    else
    {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }
    int transpose = (major > 1000) || (minor > 1000);

    int i;
    for (i = 0; i < net->n && i < cutoff; ++i)
    {
        ILayer* pLayer = net->jjLayers[i];
        if (pLayer->getLayer()->dontload)
            continue;

        if (pLayer->getType() == CONVOLUTIONAL)
        {
            ConvolutionLayer* pConvLayer = (ConvolutionLayer*)pLayer;
            pConvLayer->load_convolutional_weights_cpu(fp);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
    return true;
}


JJ::network* Detector::readConfigFile(const char* filename, int batch)
{
    IniParser parser;
    if (!parser.LoadFromFile(filename))
        return nullptr;
    
    JJ::network* pNetWork = new JJ::network;
    pNetWork->n = parser.GetSectionCount() - 1; //  LayerData count
    pNetWork->seen = (uint64_t*)calloc(1, sizeof(uint64_t));
    
    

    size_params params;

    if (!parseNetOptions(&parser, pNetWork))
        return nullptr;

    params.h = pNetWork->h;
    params.w = pNetWork->w;
    params.c = pNetWork->c;
    params.inputs = pNetWork->inputs;
    if (batch > 0)
        pNetWork->batch = batch;
    params.batch = pNetWork->batch;
    params.time_steps = pNetWork->time_steps;
    params.net = pNetWork;



    size_t workspace_size = 0;    

    // read convolutional, maxpool
    for(int i=0; i<pNetWork->n; i++)
    {
        int sectionIndex = i + 1;
        params.index = i;
        fprintf(stderr, "%5d ", i);
        std::string sectionName = parser.GetSectionByIndex(sectionIndex);
        int itemCnt = parser.GetSectionItemCount(sectionIndex);

        JJ::ILayer* pLayer = nullptr;
        JJ::LAYER_TYPE lt = string_to_layer_type(sectionName.c_str());

        // [convolutional]
        if (lt == JJ::CONVOLUTIONAL)
        {
            //l = parse_convolutional(options, params);
            pLayer = new ConvolutionLayer;
        }
        //else if (lt == REGION) {
            //l = parse_region(options, params);
        //}
        else if (lt == YOLO) {
            //l = parse_yolo(options, params);
            pLayer = new YoloLayer;
        }
        //else if (lt == SOFTMAX) {
            //l = parse_softmax(options, params);
            //net.hierarchy = l.softmax_tree;
        //}
        else if (lt == MAXPOOL) {
            //l = parse_maxpool(options, params);
            pLayer = new MaxpoolLayer;
        }
        //else if (lt == REORG) {
            //l = parse_reorg(options, params);
        //}
        else if (lt == ROUTE) {
            //l = parse_route(options, params, net);
            pLayer = new RouteLayer;
        }
        else if (lt == UPSAMPLE) {
            //l = parse_upsample(options, params, net);
            pLayer = new UpsampleLayer;
        }
        //else if (lt == SHORTCUT) {
            //l = parse_shortcut(options, params, net);
        //}
        else {
            fprintf(stderr, "Type not recognized: %s\n", sectionName.c_str());
            return nullptr;
        }

        if (!pLayer->load(&parser, sectionIndex, params))
            return nullptr;

        LayerData* pLayerInfo = pLayer->getLayer();
        pLayerInfo->dontload = parser.ReadInteger(sectionIndex, "dontload", 0);
        pLayerInfo->dontloadscales = parser.ReadInteger(sectionIndex, "dontloadscales", 0);
        //option_unused(options);

        // save this LayerData
        pNetWork->jjLayers.push_back(pLayer);

        if (pLayerInfo->workspace_size > workspace_size)
            workspace_size = pLayerInfo->workspace_size;

        {
            params.h = pLayerInfo->out_h;
            params.w = pLayerInfo->out_w;
            params.c = pLayerInfo->out_c;
            params.inputs = pLayerInfo->outputs;
        }
    }


    //free_list(sections);
    pNetWork->outputs = NetWork::get_network_output_size(pNetWork);
    pNetWork->output = NetWork::get_network_output(pNetWork);
    if (workspace_size)
    {
        //printf("%ld\n", workspace_size);
        pNetWork->workspace = (float*)calloc(workspace_size, sizeof(float));
    }
    return pNetWork;
}


JJ::LAYER_TYPE Detector::string_to_layer_type(const std::string& type)
{
    if (type == "yolo") return JJ::YOLO;
    if (type == "region") return JJ::REGION;
    if (type ==  "conv" || type == "convolutional") return JJ::CONVOLUTIONAL;
    if (type == "net"  || type == "network") return JJ::NETWORK;
    if (type == "max"|| type == "maxpool") return JJ::MAXPOOL;
    if (type == "reorg") return JJ::REORG;
    if (type == "upsample") return JJ::UPSAMPLE;
    if (type == "shortcut") return JJ::SHORTCUT;
    if (type == "soft" || type == "softmax") return JJ::SOFTMAX;
    if (type == "route") return JJ::ROUTE;
    return JJ::BLANK;
}

bool Detector::parseNetOptions(const IniParser* pIniParser, JJ::network* net)
{
    int netSectionIndex = 0;
    net->batch = pIniParser->ReadInteger(netSectionIndex, "batch", 1);
    int subdivs = pIniParser->ReadInteger(netSectionIndex, "subdivisions", 1);
    net->time_steps = pIniParser->ReadInteger(netSectionIndex, "time_steps", 1);
    net->batch /= subdivs;
    net->batch *= net->time_steps;

    net->adam = pIniParser->ReadInteger(netSectionIndex, "adam", 0);
    if (net->adam)
    {
        net->B1 = pIniParser->ReadFloat(netSectionIndex, "B1", .9);
        net->B2 = pIniParser->ReadFloat(netSectionIndex, "B2", .999);
    }

    net->h = pIniParser->ReadInteger(netSectionIndex, "height", 0);
    net->w = pIniParser->ReadInteger(netSectionIndex, "width", 0);
    net->c = pIniParser->ReadInteger(netSectionIndex, "channels", 0);
    net->inputs = pIniParser->ReadInteger(netSectionIndex, "inputs", net->h * net->w * net->c);

    if (!net->inputs && !(net->h && net->w && net->c))
        return false;
    return true;
}


// detect on CPU
void Detector::network_predict_cpu(network* net, float *input)
{
     float* pInput = input;
     for (int i = 0; i < net->n; ++i)
     {
         ILayer* pLayer = net->jjLayers[i];
         pLayer->forward_layer_cpu(net, pInput, 0);
         pInput = pLayer->getLayer()->output;
     }
}

NS_JJ_END