#include "route_layer.h"

NS_JJ_BEGIN

LayerData RouteLayer::make_route_layer(int batch, int n, std::vector<int> input_layers, std::vector<int> input_sizes)
{
    fprintf(stderr, "route ");
    LayerData l = { 0 };
    l.batch = batch;
    l.n = n;
    m_input_layers = input_layers;
    m_input_sizes = input_sizes;
    int i;
    int outputs = 0;
    for (i = 0; i < n; ++i) {
        fprintf(stderr, " %d", input_layers[i]);
        outputs += input_sizes[i];
    }
    fprintf(stderr, "\n");
    l.outputs = outputs;
    l.inputs = outputs;
    l.output = (float*)calloc(outputs*batch, sizeof(float));
    //l.output_int8.assign(outputs*batch, 0);
    return l;
}

bool RouteLayer::load(const IniParser* pParser, int section, size_params params)
{
    std::string l = pParser->ReadString(section, "layers");
    std::vector<int> results;
    StringUtil::splitInt(results, l, ",");
    int nSize = results.size();
    std::vector<int> layers;
    layers.assign(nSize, 0);
    std::vector<int> sizes;
    sizes.assign(nSize, 0);

    for (int i = 0; i < nSize; i++)
    {
        int index = results[i];
        if (index < 0)
            index = params.index + index;

        layers[i] = index;
        sizes[i] = params.net->jjLayers[index]->getLayer()->outputs;
    }

    int batch = params.batch;

    m_ld = make_route_layer(batch, nSize, layers, sizes);
    setType(ROUTE);

    JJ::LayerData* first = params.net->jjLayers[layers[0]]->getLayer();
    m_ld.out_w = first->out_w;
    m_ld.out_h = first->out_h;
    m_ld.out_c = first->out_c;
    for (int i = 1; i < nSize; ++i)
    {
        int index = layers[i];
        JJ::LayerData* next = params.net->jjLayers[index]->getLayer();
        if (next->out_w == first->out_w && next->out_h == first->out_h)
        {
            m_ld.out_c += next->out_c;
        }
        else
        {
            m_ld.out_h = m_ld.out_w = m_ld.out_c = 0;
        }
    }
    //return LayerData;
    return true;
}



void RouteLayer::forward_layer_cpu(JJ::network* pNet, float *input, int train)
{
    LayerData& l = m_ld;

    int i, j;
    int offset = 0;
    // number of merged layers
    for (i = 0; i < l.n; ++i) {
        int index = m_input_layers[i];                    // source LayerData index
        ILayer* pLayer = pNet->jjLayers[index];
        float *input = pLayer->getLayer()->output;    // source LayerData output ptr
        int input_size = m_input_sizes[i];                // source LayerData size
                                                        // batch index
        for (j = 0; j < l.batch; ++j) {
            memcpy(l.output + offset + j * l.outputs, input + j * input_size, input_size * sizeof(float));
        }
        offset += input_size;
    }
}
NS_JJ_END