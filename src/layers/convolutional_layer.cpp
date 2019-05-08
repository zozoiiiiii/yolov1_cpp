#include "convolutional_layer.h"

NS_JJ_BEGIN
static inline ACTIVATION get_activation(const std::string& s)
{
    if (s == "logistic") return LOGISTIC;
    if (s == "loggy") return LOGGY;
    if (s == "relu") return RELU;
    if (s ==  "elu") return ELU;
    if (s == "relie") return RELIE;
    if (s == "plse") return PLSE;
    if (s ==  "hardtan") return HARDTAN;
    if (s == "lhtan") return LHTAN;
    if (s == "linear") return LINEAR;
    if (s ==  "ramp") return RAMP;
    if (s ==  "leaky") return LEAKY;
    if (s ==  "tanh") return TANH;
    if (s == "stair") return STAIR;
    //fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}


static inline float stair_activate(float x)
{
    int n = floor(x);
    if (n % 2 == 0) return floor(x / 2.);
    else return (x - n) + floor(x / 2.);
}
static inline float hardtan_activate(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
static inline float linear_activate(float x) { return x; }
static inline float logistic_activate(float x) { return 1. / (1. + exp(-x)); }
static inline float loggy_activate(float x) { return 2. / (1. + exp(-x)) - 1; }
static inline float relu_activate(float x) { return x * (x > 0); }
static inline float elu_activate(float x) { return (x >= 0)*x + (x < 0)*(exp(x) - 1); }
static inline float relie_activate(float x) { return (x > 0) ? x : .01*x; }
static inline float ramp_activate(float x) { return x * (x > 0) + .1*x; }
static inline float leaky_activate(float x) { return (x > 0) ? x : .1*x; }
static inline float tanh_activate(float x) { return (exp(2 * x) - 1) / (exp(2 * x) + 1); }
static inline float plse_activate(float x)
{
    if (x < -4) return .01 * (x + 4);
    if (x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}

static inline float lhtan_activate(float x)
{
    if (x < 0) return .001*x;
    if (x > 1) return .001*(x - 1) + 1;
    return x;
}


static float activate(float x, ACTIVATION a)
{
    switch (a) {
    case LINEAR:
        return linear_activate(x);
    case LOGISTIC:
        return logistic_activate(x);
    case LOGGY:
        return loggy_activate(x);
    case RELU:
        return relu_activate(x);
    case ELU:
        return elu_activate(x);
    case RELIE:
        return relie_activate(x);
    case RAMP:
        return ramp_activate(x);
    case LEAKY:
        return leaky_activate(x);
    case TANH:
        return tanh_activate(x);
    case PLSE:
        return plse_activate(x);
    case STAIR:
        return stair_activate(x);
    case HARDTAN:
        return hardtan_activate(x);
    case LHTAN:
        return lhtan_activate(x);
    }
    return 0;
}

void ConvolutionLayer::activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    for (i = 0; i < n; ++i) {
        x[i] = activate(x[i], a);
    }
}


bool ConvolutionLayer::load(const IniParser* pParser, int section, size_params params)
{
    if (!(params.h && params.w && params.c))
        return false;// ("Layer before convolutional LayerData must output image.");


    m_ld.n = pParser->ReadInteger(section, "filters", 1);
    m_ld.size = pParser->ReadInteger(section, "size", 1);
    m_ld.stride = pParser->ReadInteger(section, "stride", 1);
    m_ld.pad = pParser->ReadInteger(section, "padding", 0);
    int pad = pParser->ReadInteger(section, "pad", 0);
    if (pad)
        m_ld.pad = m_ld.size / 2;

    std::string activation_s = pParser->ReadString(section, "activation", "logistic");
    m_conv.activation = get_activation(activation_s);

    m_ld.h = params.h;
    m_ld.w = params.w;
    m_ld.c = params.c;
    m_ld.batch = params.batch;

    m_conv.batch_normalize = pParser->ReadInteger(section, "batch_normalize", 0);
    m_conv.binary = pParser->ReadInteger(section, "binary", 0);
    m_conv.xnor = pParser->ReadInteger(section, "xnor", 0);
    m_conv.use_bin_output = pParser->ReadInteger(section, "bin_output", 0);

    make_convolutional_layer();

    m_conv.flipped = pParser->ReadInteger(section, "flipped", 0);
;
    return true;
}


float rand_uniform(float min, float max)
{
    if (max < min)
    {
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)rand() / RAND_MAX * (max - min)) + min;
}


int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2 * l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2 * l.pad - l.size) / l.stride + 1;
}


size_t ConvolutionLayer::get_workspace_size()
{
    LayerData& l = m_ld;
    if (m_conv.xnor)
        return (size_t)m_conv.bit_align*l.size*l.size*l.c * sizeof(float);

    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c * sizeof(float);
}

void ConvolutionLayer::make_convolutional_layer()
{
    int i;
    
    setType(CONVOLUTIONAL);
    //m_ld.quantized = quantized;

    int weightsSize = m_ld.c * m_ld.n * m_ld.size*m_ld.size;
    m_weight.weights = (float*)calloc(weightsSize, sizeof(float));
    m_weight.biases = (float*)calloc(m_ld.n, sizeof(float));

    float scale = sqrt(2. / (m_ld.size * m_ld.size * m_ld.c));
    for (i = 0; i < weightsSize; ++i)
        m_weight.weights[i] = scale * rand_uniform(-1, 1) ;

    int out_h = convolutional_out_height(m_ld);
    int out_w = convolutional_out_width(m_ld);
    m_ld.out_h = out_h;
    m_ld.out_w = out_w;
    m_ld.out_c = m_ld.n;
    m_ld.outputs = m_ld.out_h * m_ld.out_w * m_ld.out_c;
    m_ld.inputs = m_ld.w * m_ld.h * m_ld.c;

    m_ld.output = (float*)calloc(m_ld.batch * m_ld.outputs, sizeof(float));

    if (m_conv.binary)
    {
        m_weight.binary_weights = (float*)calloc (weightsSize, sizeof(float));
        m_weight.scales = (float*)calloc(m_ld.n, sizeof(float));
    }

    if (m_conv.xnor)
    {
        m_weight.binary_weights = (float*)calloc(weightsSize, sizeof(float));
        m_weight.binary_input = (float*)calloc(m_ld.inputs*m_ld.batch, sizeof(float));

        int align = 32;// 8;
        int src_align = m_ld.out_h*m_ld.out_w;
        m_conv.bit_align = src_align + (align - src_align % align);

        m_conv.mean_arr = (float*)calloc(m_ld.n, sizeof(float));
    }

    if (m_conv.batch_normalize)
    {
        m_weight.scales = (float*)calloc(m_ld.n, sizeof(float));
        for (i = 0; i < m_ld.n; ++i)
        {
            m_weight.scales[i] = 1;
        }


        m_weight.rolling_mean = (float*)calloc(m_ld.n, sizeof(float));
        m_weight.rolling_variance = (float*)calloc(m_ld.n, sizeof(float));
    }


    m_ld.workspace_size = get_workspace_size();

    m_conv.bflops = (2.0 * m_ld.n * m_ld.size*m_ld.size*m_ld.c * m_ld.out_h*m_ld.out_w) / 1000000000.;
    if (m_conv.xnor && m_conv.use_bin_output) fprintf(stderr, "convXB");
    else if (m_conv.xnor) fprintf(stderr, "convX ");
    else fprintf(stderr, "conv  ");

    fprintf(stderr, "%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d %5.3f BF\n",
        m_ld.n, m_ld.size, m_ld.size, m_ld.stride, m_ld.w, m_ld.h, m_ld.c,
        m_ld.out_w, m_ld.out_h, m_ld.out_c, m_conv.bflops);
}







void ConvolutionLayer::fuse_batchnorm()
{
    LayerData* l = getLayer();
    if (getConv()->batch_normalize)
    {
        int f;
        for (f = 0; f < l->n; ++f)
        {
            ConvolutionWeight* pWeight = getWeight();
            pWeight->biases[f] = pWeight->biases[f] - pWeight->scales[f] * pWeight->rolling_mean[f] / (sqrtf(pWeight->rolling_variance[f]) + .000001f);

            const size_t filter_size = l->size*l->size*l->c;
            int i;
            for (i = 0; i < filter_size; ++i)
            {
                int w_index = f * filter_size + i;

                pWeight->weights[w_index] = pWeight->weights[w_index] * pWeight->scales[f] / (sqrtf(pWeight->rolling_variance[f]) + .000001f);
            }
        }

        getConv()->batch_normalize = 0;
    }
}


//////////////////////////////////forward//////////////////////////////////////////
void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for (f = 0; f < n; ++f) {
        float mean = 0;
        for (i = 0; i < size; ++i) {
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for (i = 0; i < size; ++i) {
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for (i = 0; i < n; ++i) {
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void activate_array_cpu_custom(float* x, const int n, const ACTIVATION a)
{
    int i = 0;
    if (a == LINEAR) {}
    else {
        for (i = 0; i < n; ++i) {
            x[i] = activate(x[i], a);
        }
    }
}


// 32 channels -> 1 channel (with 32 floats)
// 256 channels -> 8 channels (with 32 floats)
void repack_input(float *input, float *re_packed_input, int w, int h, int c)
{
    const int items_per_channel = w * h;
    int chan, i;
    for (chan = 0; chan < c; chan += 32)
    {
        for (i = 0; i < items_per_channel; ++i)
        {
            int c_pack;
            for (c_pack = 0; c_pack < 32; ++c_pack) {
                float src = input[(chan + c_pack)*items_per_channel + i];

                re_packed_input[chan*items_per_channel + i * 32 + c_pack] = src;
            }
        }
    }
}


// im2col.c
float im2col_get_pixel(float *im, int height, int width, int channels,
    int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width * (row + height * channel)];
}

// im2col.c
//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                    im_row, im_col, c_im, pad);
            }
        }
    }
}


//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_custom(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col)
{
    im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col);
}



// transpose uint32_t matrix
void transpose_uint32(uint32_t *src, uint32_t *dst, int src_h, int src_w, int src_align, int dst_align)
{
    //l.bit_align - algined (n) by 32
    //new_ldb - aligned (k) by 256

    int i;
    //#pragma omp parallel for
    for (i = 0; i < src_h; i += 1)  // l.size*l.size*l.c;
    {
        int j;
        for (j = 0; j < src_w; j += 1)  // out_h*out_w;
        {
            ((uint32_t *)dst)[j*dst_align / 32 + i] = ((uint32_t *)src)[i*src_align + j];
        }
    }
}



static inline uint64_t xnor_int64(uint64_t a, uint64_t b) {
    return ~(a^b);
}


static inline int popcnt_64(uint64_t val64) {
#ifdef WIN32  // Windows
#ifdef _WIN64 // Windows 64-bit
    int tmp_count = __popcnt64(val64);
#else         // Windows 32-bit
    int tmp_count = __popcnt(val64);
    tmp_count += __popcnt(val64 >> 32);
#endif
#else   // Linux
#ifdef __x86_64__  // Linux 64-bit
    int tmp_count = __builtin_popcountll(val64);
#else  // Linux 32-bit
    int tmp_count = __builtin_popcount(val64);
    tmp_count += __builtin_popcount(val64);
#endif
#endif
    return tmp_count;
}


void gemm_nn_custom_bin_mean_transposed(int M, int N, int K, float ALPHA_UNUSED,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    int i, j, k, h;

#pragma omp parallel for
    for (i = 0; i < M; ++i) {   // l.n - filters [16 - 55 - 1024]
        float mean_val = mean_arr[i];

        for (j = 0; j < N; ++j) { // out_h*out_w - one channel output size [169 - 173056]
            int count = 0;

            for (k = 0; k < K; k += 64) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                uint64_t a_bit64 = *((uint64_t *)(A + (i*lda + k) / 8));
                uint64_t b_bit64 = *((uint64_t *)(B + (j*ldb + k) / 8));
                uint64_t c_bit64 = xnor_int64(a_bit64, b_bit64);

                int tmp_count = popcnt_64(c_bit64);

                if (K - k < 64)  tmp_count = tmp_count - (64 - (K - k));    // remove extra bits
                count += tmp_count;
                //binary_int64_printf(c_bit64);
                //printf(", count = %d \n\n", tmp_count);
            }

            C[i*ldc + j] = (2 * count - K) * mean_val;
        }
    }
}





static inline void set_bit(unsigned char *const dst, size_t index) {
    size_t dst_i = index / 8;
    int dst_shift = index % 8;
    dst[dst_i] |= 1 << dst_shift;
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_custom_bin(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col, int bit_align)
{
    int c;
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int channels_col = channels * ksize * ksize;

    // optimized version
    if (height_col == height && width_col == width && stride == 1 && pad == 1)
    {
        int new_ldb = bit_align;

#pragma omp parallel for
        for (c = 0; c < channels_col; ++c) {
            int h, w;
            int w_offset = c % ksize;
            int h_offset = (c / ksize) % ksize;
            int c_im = c / ksize / ksize;
            for (h = pad; h < height_col - pad; ++h) {
                for (w = pad; w < width_col - pad - 8; w += 1) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    float val = data_im[im_col + width * (im_row + height * c_im)];
                    if (val > 0) set_bit((unsigned char* const)data_col, col_index);
                }

                for (; w < width_col - pad; ++w) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                    float val = data_im[im_col + width * (im_row + height * c_im)];
                    if (val > 0) set_bit((unsigned char* const)data_col, col_index);
                }
            }

            {
                w = 0;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit((unsigned char* const)data_col, col_index);
                }
            }

            {
                w = width_col - 1;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit((unsigned char* const)data_col, col_index);
                }
            }

            {
                h = 0;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit((unsigned char* const)data_col, col_index);
                }
            }

            {
                h = height_col - 1;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit((unsigned char* const)data_col, col_index);
                }
            }
        }

    }
    else {
        printf("\n Error: is no non-optimized version \n");
        //im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col); // must be aligned for transpose after float_to_bin
        // float_to_bit(b, t_input, src_size);
        // transpose_bin(t_input, *t_bit_input, k, n, bit_align, new_ldb, 8);
    }
}

static inline unsigned char get_bit(unsigned char const*const src, size_t index) {
    size_t src_i = index / 8;
    int src_shift = index % 8;
    unsigned char val = (src[src_i] & (1 << src_shift)) > 0;
    return val;
}


uint8_t reverse_8_bit(uint8_t a) {
    return ((a * 0x0802LU & 0x22110LU) | (a * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16;
}


uint32_t reverse_32_bit(uint32_t a)
{
    // unsigned int __rbit(unsigned int val) // for ARM    //__asm__("rbit %0, %1\n" : "=r"(output) : "r"(input));
    return (reverse_8_bit(a >> 24) << 0) |
        (reverse_8_bit(a >> 16) << 8) |
        (reverse_8_bit(a >> 8) << 16) |
        (reverse_8_bit(a >> 0) << 24);
}

#define swap(a0, a1, j, m) t = (a0 ^ (a1 >>j)) & m; a0 = a0 ^ t; a1 = a1 ^ (t << j);

void transpose32_optimized(uint32_t A[32]) {
    int j, k;
    unsigned m, t;

    //m = 0x0000FFFF;
    //for (j = 16; j != 0; j = j >> 1, m = m ^ (m << j)) {
    //    for (k = 0; k < 32; k = (k + j + 1) & ~j) {
    //        t = (A[k] ^ (A[k + j] >> j)) & m;
    //        A[k] = A[k] ^ t;
    //        A[k + j] = A[k + j] ^ (t << j);
    //    }
    //}

    j = 16;
    m = 0x0000FFFF;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 8;
    m = 0x00ff00ff;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 4;
    m = 0x0f0f0f0f;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 2;
    m = 0x33333333;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 1;
    m = 0x55555555;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    // reverse Y
    for (j = 0; j < 16; ++j) {
        uint32_t tmp = A[j];
        A[j] = reverse_32_bit(A[31 - j]);
        A[31 - j] = reverse_32_bit(tmp);
    }
}

void transpose_32x32_bits_reversed_diagonale(uint32_t *A, uint32_t *B, int m, int n)
{
    unsigned A_tmp[32];
    int i;
#pragma unroll
    for (i = 0; i < 32; ++i) A_tmp[i] = A[i * m];
    transpose32_optimized(A_tmp);
#pragma unroll
    for (i = 0; i < 32; ++i) B[i*n] = A_tmp[i];
}


// transpose by 32-bit
void transpose_bin(uint32_t *A, uint32_t *B, const int n, const int m,
    const int lda, const int ldb, const int block_size)
{
    int i;
#pragma omp parallel for
    for (i = 0; i < n; i += 32) {
        int j;
        for (j = 0; j < m; j += 32) {
            int a_index = i * lda + j;
            int b_index = j * ldb + i;
            transpose_32x32_bits_reversed_diagonale(&A[a_index / 32], &B[b_index / 32], lda / 32, ldb / 32);
            //transpose_32x32_bits_my(&A[a_index/32], &B[b_index/32], lda/32, ldb/32);
        }
        for (; j < m; ++j)
        {
            if (get_bit((const unsigned char*)A, i*lda + j)) set_bit((unsigned char* const)B, j*ldb + i);
        }
    }
}

// binary transpose
size_t binary_transpose_align_input(int k, int n, float *b, char **t_bit_input, size_t ldb_align, int bit_align)
{
    size_t new_ldb = k + (ldb_align - k % ldb_align); // (k / 8 + 1) * 8;
    size_t t_intput_size = new_ldb * bit_align;// n;
    size_t t_bit_input_size = t_intput_size / 8;// +1;
    *t_bit_input = (char*)calloc(t_bit_input_size, sizeof(char));

    //printf("\n t_bit_input_size = %d, k = %d, n = %d, new_ldb = %d \n", t_bit_input_size, k, n, new_ldb);
    int src_size = k * bit_align;
    transpose_bin((uint32_t*)b, (uint32_t*)*t_bit_input, k, n, bit_align, new_ldb, 8);

    return t_intput_size;
}

void gemm_nn(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i)
    {
        for (k = 0; k < K; ++k)
        {
            register float A_PART = ALPHA * A[i*lda + k];
            for (j = 0; j < N; ++j)
            {
                C[i*ldc + j] += A_PART * B[k*ldb + j];
            }
        }
    }
}

void float_to_bit(float *src, unsigned char *dst, size_t size)
{
    size_t dst_size = size / 8 + 1;
    memset(dst, 0, dst_size);

    size_t i;
    char *byte_arr = (char*)calloc(size, sizeof(char));
    for (i = 0; i < size; ++i) {
        if (src[i] > 0) byte_arr[i] = 1;
    }

    //for (i = 0; i < size; ++i) {
    //    dst[i / 8] |= byte_arr[i] << (i % 8);
    //}

    for (i = 0; i < size; i += 8) {
        char dst_tmp = 0;
        dst_tmp |= byte_arr[i + 0] << 0;
        dst_tmp |= byte_arr[i + 1] << 1;
        dst_tmp |= byte_arr[i + 2] << 2;
        dst_tmp |= byte_arr[i + 3] << 3;
        dst_tmp |= byte_arr[i + 4] << 4;
        dst_tmp |= byte_arr[i + 5] << 5;
        dst_tmp |= byte_arr[i + 6] << 6;
        dst_tmp |= byte_arr[i + 7] << 7;
        dst[i / 8] = dst_tmp;
    }
    free(byte_arr);
}



// output = active * ( input vector * weight matrix + bias)
void ConvolutionLayer::forward_layer_cpu(JJ::network* pNet, float *input, int train)
{
    LayerData& l = m_ld;
    int out_h = (l.h + 2 * l.pad - l.size) / l.stride + 1;    // output_height=input_height for stride=1 and pad=1
    int out_w = (l.w + 2 * l.pad - l.size) / l.stride + 1;    // output_width=input_width for stride=1 and pad=1
    int i, f, j;

    // fill zero (ALPHA)
    for (i = 0; i < l.outputs*l.batch; ++i)
        l.output[i] = 0;

    // 是否支持 二值化卷积网络(xnor-net)
    if (m_conv.xnor)
    {
        if (!m_conv.align_bit_weights)
        {
            // 把+,-,x操作简化成+,- 
            binarize_weights(m_weight.weights, l.n, l.c*l.size*l.size, m_weight.binary_weights);
            //printf("\n binarize_weights l.align_bit_weights = %p \n", l.align_bit_weights);
        }
        binarize_cpu(input, l.c*l.h*l.w*l.batch, m_weight.binary_input);

        m_weight.weights = m_weight.binary_weights;
        input = m_weight.binary_input;
    }

    // l.n - number of filters on this LayerData
    // l.c - channels of input-array
    // l.h - height of input-array
    // l.w - width of input-array
    // l.size - width and height of filters (the same size for all filters)


    // 1. Convolution !!! 图片用卷积核矩阵来过滤

    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h * out_w;
    float *a = m_weight.weights;
    float *b = pNet->workspace;
    float *c = l.output;

    /*
    1. im2col: use im2col to change the input into a new array, used for filter (https://blog.csdn.net/mrhiuser/article/details/52672824)
    2. gemm: im2col result * matrix from weight = output
    */

    // convolution as GEMM (as part of BLAS)
    for (i = 0; i < l.batch; ++i)
    {
        //im2col_cpu(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);    // im2col.c
        //im2col_cpu_custom(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);    // AVX2

        // XNOR-net - bit-1: weights, input, calculation
        if (m_conv.xnor && m_conv.align_bit_weights && (l.stride == 1 && l.pad == 1))
        {
            memset(b, 0, m_conv.bit_align*l.size*l.size*l.c * sizeof(float));

            if (l.c % 32 == 0)
            {
                //printf(" l.index = %d - new XNOR \n", l.index);

                int ldb_align = m_conv.lda_align;
                size_t new_ldb = k + (ldb_align - k % ldb_align); // (k / 8 + 1) * 8;
                size_t t_intput_size = new_ldb * m_conv.bit_align;// n;
                size_t t_bit_input_size = t_intput_size / 8;// +1;

                const int new_c = l.c / 32;

                float *re_packed_input = (float*)calloc(l.c * l.w * l.h, sizeof(float));
                uint32_t *bin_re_packed_input = (uint32_t*)calloc(new_c * l.w * l.h + 1, sizeof(uint32_t));

                // float32x4 by channel (as in cuDNN)
                repack_input(input, re_packed_input, l.w, l.h, l.c);

                // 32 x floats -> 1 x uint32_t
                float_to_bit(re_packed_input, (unsigned char *)bin_re_packed_input, l.c * l.w * l.h);

                free(re_packed_input);

                im2col_cpu_custom((float *)bin_re_packed_input, new_c, l.h, l.w, l.size, l.stride, l.pad, b);
                //im2col_cpu((float *)bin_re_packed_input, new_c, l.h, l.w, l.size, l.stride, l.pad, b);

                free(bin_re_packed_input);

                int new_k = l.size*l.size*l.c / 32;


                char *t_bit_input = (char*)calloc(t_bit_input_size, sizeof(char));

                transpose_uint32((uint32_t *)b, (uint32_t*)t_bit_input, new_k, n, n, new_ldb);

                // the main GEMM function
                gemm_nn_custom_bin_mean_transposed(m, n, k, 1, (unsigned char*)m_conv.align_bit_weights, new_ldb,
                    (unsigned char*)t_bit_input, new_ldb, c, n, m_conv.mean_arr);


                free(t_bit_input);

            }
            else { // else (l.c % 32 != 0)

            //im2col_cpu_custom_align(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b, l.bit_align);
                im2col_cpu_custom_bin(input, l.c, l.h, l.w, l.size, l.stride, l.pad, b, m_conv.bit_align);

                int ldb_align = m_conv.lda_align;
                size_t new_ldb = k + (ldb_align - k % ldb_align);
                char *t_bit_input = NULL;
                size_t t_intput_size = binary_transpose_align_input(k, n, b, &t_bit_input, ldb_align, m_conv.bit_align);

                // 5x times faster than gemm()-float32
                gemm_nn_custom_bin_mean_transposed(m, n, k, 1, (unsigned char*)m_conv.align_bit_weights, new_ldb,
                    (unsigned char*)t_bit_input, new_ldb, c, n, m_conv.mean_arr);

                //gemm_nn_custom_bin_mean_transposed(m, n, k, 1, bit_weights, k, t_bit_input, new_ldb, c, n, mean_arr);

                //free(t_input);
                free(t_bit_input);
            }
        }
        else {
            // from w,h,c, get vector b
            im2col_cpu_custom(input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);    // AVX2
            int t;
#pragma omp parallel for
            for (t = 0; t < m; ++t) // how many filters, get the depth
            {
                // every filter
                gemm_nn(1, n, k, 1, a + t * k, k, b, n, c + t * n, n);
            }
        }
        c += n * m;
        input += l.c*l.h*l.w;

    }


    int const out_size = out_h * out_w;

    // 2. Batch normalization 批量归一化， let the output value between 0-1, then every LayerData can have the same range.
    if (m_conv.batch_normalize)
    {
        int b;
        for (b = 0; b < l.batch; b++)
        {
            for (f = 0; f < l.out_c; ++f)
            {
                for (i = 0; i < out_size; ++i)
                {
                    int index = f * out_size + i;
                    l.output[index + b * l.outputs] = (l.output[index + b * l.outputs] - m_weight.rolling_mean[f]) / (sqrtf(m_weight.rolling_variance[f]) + .000001f);
                }
            }

            // scale_bias
            for (i = 0; i < l.out_c; ++i) {
                for (j = 0; j < out_size; ++j) {
                    l.output[i*out_size + j + b * l.outputs] *= m_weight.scales[i];
                }
            }
        }
    }

    // 3. Add BIAS
    //if (l.batch_normalize)
    {
        for (int b = 0; b < l.batch; b++)
        for (int i = 0; i < l.n; ++i)
        for (int j = 0; j < out_size; ++j)
             l.output[i*out_size + j + b * l.outputs] += m_weight.biases[i];
    }

    // 4. Activation function (LEAKY or LINEAR),   make the output value range into 0-1
    //if (l.activation == LEAKY) {
    //    for (i = 0; i < l.n*out_size; ++i) {
    //        l.output[i] = leaky_activate(l.output[i]);
    //    }
    //}
    //activate_array_cpu_custom(l.output, l.n*out_size, l.activation);
    activate_array_cpu_custom(l.output, l.outputs*l.batch, m_conv.activation);

}






void get_mean_array(float *src, size_t size, size_t filters, float *mean_arr) {
    size_t i, counter;
    counter = 0;
    for (i = 0; i < size; i += size / filters) {
        mean_arr[counter++] = fabs(src[i]);
    }
}

void ConvolutionLayer::binary_align_weights()
{
    LayerData* l = &m_ld;

    int m = l->n;
    int k = l->size*l->size*l->c;
    size_t new_lda = k + (m_conv.lda_align - k % m_conv.lda_align); // (k / 8 + 1) * 8;
    //l->new_lda = new_lda;

    binarize_weights(m_weight.weights, m, k, m_weight.binary_weights);

    size_t align_weights_size = new_lda * m;
    m_conv.align_bit_weights_size = align_weights_size / 8 + 1;
    float *align_weights = (float*)calloc(align_weights_size, sizeof(float));
    m_conv.align_bit_weights = (char*)calloc(m_conv.align_bit_weights_size, sizeof(char));

    size_t i, j;
    // align A without transpose
    for (i = 0; i < m; ++i) {
        for (j = 0; j < k; ++j) {
            align_weights[i*new_lda + j] = m_weight.binary_weights[i*k + j];
        }
    }


    //if (l->c % 32 == 0)
    int gpu_index = 0;
    if (gpu_index < 0 && l->stride == 1 && l->pad == 1 && l->c % 32 == 0)
    {
        int fil, chan;
        const int items_per_filter = l->c * l->size * l->size;
        //const int dst_items_per_filter = new_lda;
        for (fil = 0; fil < l->n; ++fil)
        {
            for (chan = 0; chan < l->c; chan += 32)
            {
                const int items_per_channel = l->size*l->size;
                for (i = 0; i < items_per_channel; ++i)
                {
                    uint32_t val = 0;
                    int c_pack;
                    for (c_pack = 0; c_pack < 32; ++c_pack) {
                        float src = m_weight.binary_weights[fil*items_per_filter + (chan + c_pack)*items_per_channel + i];

                        //align_weights[fil*items_per_filter + chan*items_per_channel + i * 32 + c_pack] = src;

                        align_weights[fil*new_lda + chan * items_per_channel + i * 32 + c_pack] = src;
                        //val |= (src << c);
                    }

                }
            }
        }

        //printf("\n l.index = %d \t aw[0] = %f, aw[1] = %f, aw[2] = %f, aw[3] = %f \n", l->index, align_weights[0], align_weights[1], align_weights[2], align_weights[3]);
        //memcpy(l->binary_weights, align_weights, (l->size * l->size * l->c * l->n) * sizeof(float));

        float_to_bit(align_weights, (unsigned char*)m_conv.align_bit_weights, align_weights_size);

        get_mean_array(m_weight.binary_weights, m*k, l->n, m_conv.mean_arr);
        //get_mean_array(l->binary_weights, m*new_lda, l->n, l->mean_arr);
    }
    else
    {
        float_to_bit(align_weights, (unsigned char*)m_conv.align_bit_weights, align_weights_size);

        get_mean_array(m_weight.binary_weights, m*k, l->n, m_conv.mean_arr);
    }

    free(align_weights);
}





void ConvolutionLayer::load_convolutional_weights_cpu(FILE *fp)
{
    LayerData* pLayerInfo = &m_ld;
    ConvolutionWeight* pWeight = &m_weight;
    int num = pLayerInfo->n * pLayerInfo->c * pLayerInfo->size * pLayerInfo->size;
    int n = pLayerInfo->n;
    fread(pWeight->biases, sizeof(float), n, fp);
    if (m_conv.batch_normalize && (!pLayerInfo->dontloadscales))
    {
        fread(pWeight->scales, sizeof(float), n, fp);
        fread(pWeight->rolling_mean, sizeof(float), n, fp);
        fread(pWeight->rolling_variance, sizeof(float), n, fp);
    }
    fread(pWeight->weights, sizeof(float), num, fp);
}

NS_JJ_END