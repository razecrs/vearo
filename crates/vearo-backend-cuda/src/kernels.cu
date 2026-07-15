// Vearo CUDA Reference Kernels

// Binary elementwise helper to map 1D index to LHS and RHS offsets via broadcasting
__device__ void get_binary_offsets(
    int idx, const int* info, int& l_off, int& r_off
) {
    int out_rank = info[0];
    int l_rank = info[1];
    int r_rank = info[2];

    const int* out_dims = info + 3;
    const int* l_dims = info + 19;
    const int* l_strides = info + 27;
    const int* r_dims = info + 35;
    const int* r_strides = info + 43;

    int remaining = idx;
    l_off = 0;
    r_off = 0;

    for (int i = out_rank - 1; i >= 0; --i) {
        int coord = remaining % out_dims[i];
        remaining /= out_dims[i];

        if (i >= out_rank - l_rank) {
            int l_dim_idx = i - (out_rank - l_rank);
            if (l_dims[l_dim_idx] > 1) {
                l_off += coord * l_strides[l_dim_idx];
            }
        }
        if (i >= out_rank - r_rank) {
            int r_dim_idx = i - (out_rank - r_rank);
            if (r_dims[r_dim_idx] > 1) {
                r_off += coord * r_strides[r_dim_idx];
            }
        }
    }
}

// Binary elementwise operations with full multidimensional broadcasting support
extern "C" __global__ void add_broadcast_kernel(
    const float* lhs, const float* rhs, float* out,
    const int* info, int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    int l_off, r_off;
    get_binary_offsets(idx, info, l_off, r_off);
    out[idx] = lhs[l_off] + rhs[r_off];
}

extern "C" __global__ void sub_broadcast_kernel(
    const float* lhs, const float* rhs, float* out,
    const int* info, int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    int l_off, r_off;
    get_binary_offsets(idx, info, l_off, r_off);
    out[idx] = lhs[l_off] - rhs[r_off];
}

extern "C" __global__ void mul_broadcast_kernel(
    const float* lhs, const float* rhs, float* out,
    const int* info, int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    int l_off, r_off;
    get_binary_offsets(idx, info, l_off, r_off);
    out[idx] = lhs[l_off] * rhs[r_off];
}

extern "C" __global__ void div_broadcast_kernel(
    const float* lhs, const float* rhs, float* out,
    const int* info, int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    int l_off, r_off;
    get_binary_offsets(idx, info, l_off, r_off);
    out[idx] = lhs[l_off] / rhs[r_off];
}

// Unary operations
extern "C" __global__ void relu_forward(const float* x, float* out, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
    }
}

extern "C" __global__ void relu_backward(const float* x, const float* grad_out, float* grad_in, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        grad_in[idx] = x[idx] > 0.0f ? grad_out[idx] : 0.0f;
    }
}

extern "C" __global__ void gelu_forward(const float* x, float* out, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float val = x[idx];
        float tanh_in = 0.79788456f * (val + 0.044715f * val * val * val);
        out[idx] = 0.5f * val * (1.0f + tanhf(tanh_in));
    }
}

extern "C" __global__ void gelu_backward(const float* x, const float* grad_out, float* grad_in, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float val = x[idx];
        float val3 = val * val * val;
        float inner = 0.79788456f * (val + 0.044715f * val3);
        float t = tanhf(inner);
        float sech_sq = 1.0f - t * t;
        float inner_deriv = 0.79788456f * (1.0f + 0.134145f * val * val);
        float gelu_deriv = 0.5f * (1.0f + t) + 0.5f * val * sech_sq * inner_deriv;
        grad_in[idx] = grad_out[idx] * gelu_deriv;
    }
}

// Reductions helper
__device__ void get_reduce_offset(
    int idx, const int* info, int r_coord, int& x_off
) {
    int out_rank = info[0];
    int x_rank = info[1];
    int reduce_dim = info[2];

    const int* out_dims = info + 3;
    const int* x_strides = info + 27;

    int remaining = idx;
    int out_coord[8];
    for (int i = out_rank - 1; i >= 0; --i) {
        out_coord[i] = remaining % out_dims[i];
        remaining /= out_dims[i];
    }

    x_off = 0;
    for (int i = 0; i < x_rank; ++i) {
        int coord = (i == reduce_dim) ? r_coord : out_coord[i];
        x_off += coord * x_strides[i];
    }
}

// Reductions
extern "C" __global__ void sum_kernel(
    const float* x, float* out,
    const int* info, int reduce_size, int numel_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel_out) return;

    float sum = 0.0f;
    for (int r = 0; r < reduce_size; ++r) {
        int x_off;
        get_reduce_offset(idx, info, r, x_off);
        sum += x[x_off];
    }
    out[idx] = sum;
}

extern "C" __global__ void mean_kernel(
    const float* x, float* out,
    const int* info, int reduce_size, int numel_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel_out) return;

    float sum = 0.0f;
    for (int r = 0; r < reduce_size; ++r) {
        int x_off;
        get_reduce_offset(idx, info, r, x_off);
        sum += x[x_off];
    }
    out[idx] = sum / (float)reduce_size;
}

// Softmax
extern "C" __global__ void softmax_forward(
    const float* x, float* out,
    const int* info,
    int reduce_dim, int reduce_size, int outer_numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_numel) return;

    int x_rank = info[0];
    const int* x_dims = info + 2;
    const int* x_strides = info + 10;

    int remaining = idx;
    int coords[8];
    for (int i = x_rank - 1; i >= 0; --i) {
        if (i == reduce_dim) continue;
        coords[i] = remaining % x_dims[i];
        remaining /= x_dims[i];
    }

    float max_val = -1e30f;
    for (int r = 0; r < reduce_size; ++r) {
        int x_off = 0;
        for (int i = 0; i < x_rank; ++i) {
            int coord = (i == reduce_dim) ? r : coords[i];
            x_off += coord * x_strides[i];
        }
        float val = x[x_off];
        if (val > max_val) max_val = val;
    }

    float sum_exp = 0.0f;
    for (int r = 0; r < reduce_size; ++r) {
        int x_off = 0;
        for (int i = 0; i < x_rank; ++i) {
            int coord = (i == reduce_dim) ? r : coords[i];
            x_off += coord * x_strides[i];
        }
        sum_exp += expf(x[x_off] - max_val);
    }

    for (int r = 0; r < reduce_size; ++r) {
        int x_off = 0;
        for (int i = 0; i < x_rank; ++i) {
            int coord = (i == reduce_dim) ? r : coords[i];
            x_off += coord * x_strides[i];
        }
        out[x_off] = expf(x[x_off] - max_val) / sum_exp;
    }
}

extern "C" __global__ void softmax_backward(
    const float* y, const float* grad_out, float* grad_in,
    const int* info,
    int reduce_dim, int reduce_size, int outer_numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_numel) return;

    int y_rank = info[0];
    const int* y_dims = info + 2;
    const int* y_strides = info + 10;

    int remaining = idx;
    int coords[8];
    for (int i = y_rank - 1; i >= 0; --i) {
        if (i == reduce_dim) continue;
        coords[i] = remaining % y_dims[i];
        remaining /= y_dims[i];
    }

    float sum_go_y = 0.0f;
    for (int r = 0; r < reduce_size; ++r) {
        int off = 0;
        for (int i = 0; i < y_rank; ++i) {
            int coord = (i == reduce_dim) ? r : coords[i];
            off += coord * y_strides[i];
        }
        sum_go_y += grad_out[off] * y[off];
    }

    for (int r = 0; r < reduce_size; ++r) {
        int off = 0;
        for (int i = 0; i < y_rank; ++i) {
            int coord = (i == reduce_dim) ? r : coords[i];
            off += coord * y_strides[i];
        }
        grad_in[off] = y[off] * (grad_out[off] - sum_go_y);
    }
}

// LayerNorm
extern "C" __global__ void layernorm_forward(
    const float* x, const float* weight, const float* bias, float* out,
    int norm_dim, float eps, int outer_numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_numel) return;

    int base_idx = idx * norm_dim;

    float sum = 0.0f;
    for (int i = 0; i < norm_dim; ++i) {
        sum += x[base_idx + i];
    }
    float mean = sum / (float)norm_dim;

    float sum_sq = 0.0f;
    for (int i = 0; i < norm_dim; ++i) {
        float diff = x[base_idx + i] - mean;
        sum_sq += diff * diff;
    }
    float var = sum_sq / (float)norm_dim;
    float inv_std = 1.0f / sqrtf(var + eps);

    for (int i = 0; i < norm_dim; ++i) {
        float x_hat = (x[base_idx + i] - mean) * inv_std;
        out[base_idx + i] = x_hat * weight[i] + bias[i];
    }
}

extern "C" __global__ void layernorm_backward(
    const float* x, const float* weight, const float* bias, const float* grad_out,
    float* grad_x, float* grad_w, float* grad_b,
    int norm_dim, float eps, int outer_numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_numel) return;

    int base_idx = idx * norm_dim;

    float sum = 0.0f;
    for (int i = 0; i < norm_dim; ++i) {
        sum += x[base_idx + i];
    }
    float mean = sum / (float)norm_dim;

    float sum_sq = 0.0f;
    for (int i = 0; i < norm_dim; ++i) {
        float diff = x[base_idx + i] - mean;
        sum_sq += diff * diff;
    }
    float var = sum_sq / (float)norm_dim;
    float inv_std = 1.0f / sqrtf(var + eps);

    float sum_go_w = 0.0f;
    float sum_go_w_xhat = 0.0f;
    for (int i = 0; i < norm_dim; ++i) {
        float x_hat = (x[base_idx + i] - mean) * inv_std;
        float w_val = weight[i];
        float go_val = grad_out[base_idx + i];

        sum_go_w += go_val * w_val;
        sum_go_w_xhat += go_val * w_val * x_hat;

        atomicAdd(&grad_w[i], go_val * x_hat);
        atomicAdd(&grad_b[i], go_val);
    }

    for (int i = 0; i < norm_dim; ++i) {
        float x_hat = (x[base_idx + i] - mean) * inv_std;
        float w_val = weight[i];
        float go_val = grad_out[base_idx + i];

        float term1 = (float)norm_dim * go_val * w_val;
        float term2 = sum_go_w;
        float term3 = x_hat * sum_go_w_xhat;

        grad_x[base_idx + i] = (term1 - term2 - term3) * inv_std / (float)norm_dim;
    }
}

// Embedding
extern "C" __global__ void embedding_forward(
    const float* x, const float* weight, float* out,
    int vocab_size, int embedding_dim, int numel_x
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel_x) return;

    int token_id = (int)roundf(x[idx]);
    if (token_id < 0 || token_id >= vocab_size) return;

    int out_base = idx * embedding_dim;
    int weight_base = token_id * embedding_dim;

    for (int d = 0; d < embedding_dim; ++d) {
        out[out_base + d] = weight[weight_base + d];
    }
}

extern "C" __global__ void embedding_backward(
    const float* x, const float* grad_out, float* grad_w,
    int vocab_size, int embedding_dim, int numel_x
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel_x) return;

    int token_id = (int)roundf(x[idx]);
    if (token_id < 0 || token_id >= vocab_size) return;

    int go_base = idx * embedding_dim;
    int weight_base = token_id * embedding_dim;

    for (int d = 0; d < embedding_dim; ++d) {
        atomicAdd(&grad_w[weight_base + d], grad_out[go_base + d]);
    }
}

// CrossEntropy
extern "C" __global__ void cross_entropy_forward(
    const float* logits, const float* targets, float* out,
    int batch_size, int vocab_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    int target_class = (int)roundf(targets[idx]);
    if (target_class < 0 || target_class >= vocab_size) return;

    int base_logits = idx * vocab_size;

    float max_val = -1e30f;
    for (int c = 0; c < vocab_size; ++c) {
        float val = logits[base_logits + c];
        if (val > max_val) max_val = val;
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < vocab_size; ++c) {
        sum_exp += expf(logits[base_logits + c] - max_val);
    }

    float loss = max_val + logf(sum_exp) - logits[base_logits + target_class];
    out[idx] = loss;
}

extern "C" __global__ void cross_entropy_backward(
    const float* logits, const float* targets, const float* grad_out, float* grad_logits,
    int batch_size, int vocab_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    int target_class = (int)roundf(targets[idx]);
    if (target_class < 0 || target_class >= vocab_size) return;

    int base_logits = idx * vocab_size;
    float go_val = grad_out[0];

    float max_val = -1e30f;
    for (int c = 0; c < vocab_size; ++c) {
        float val = logits[base_logits + c];
        if (val > max_val) max_val = val;
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < vocab_size; ++c) {
        sum_exp += expf(logits[base_logits + c] - max_val);
    }

    for (int c = 0; c < vocab_size; ++c) {
        float prob = expf(logits[base_logits + c] - max_val) / sum_exp;
        float target_indicator = (c == target_class) ? 1.0f : 0.0f;
        grad_logits[base_logits + c] = (prob - target_indicator) / (float)batch_size * go_val;
    }
}

extern "C" __global__ void matmul_kernel(
    const float* lhs, const float* rhs, float* out,
    int M, int K, int N, int batch_size,
    int lhs_batch_stride, int rhs_batch_stride, int out_batch_stride
) {
    int b = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float val = 0.0f;
        const float* l_ptr = lhs + b * lhs_batch_stride + row * K;
        for (int k = 0; k < K; ++k) {
            val += l_ptr[k] * rhs[b * rhs_batch_stride + k * N + col];
        }
        out[b * out_batch_stride + row * N + col] = val;
    }
}

// Conv2D (NCHW input, OIHW weight), direct convolution. Params packed in p:
// [n, cin, h, w, cout, kh, kw, oh, ow, stride, padding].
// Backward uses gather kernels (one thread per output element, no atomics) so the
// accumulation order matches the CPU backend bit-for-bit.

extern "C" __global__ void conv2d_forward(
    const float* x, const float* weight, const float* bias, float* out,
    const int* p
) {
    int n = p[0], cin = p[1], h = p[2], w = p[3];
    int cout = p[4], kh = p[5], kw = p[6];
    int oh = p[7], ow = p[8], stride = p[9], padding = p[10];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * cout * oh * ow) return;

    int x_out = idx % ow;
    int y = (idx / ow) % oh;
    int co = (idx / (ow * oh)) % cout;
    int nn = idx / (ow * oh * cout);

    float acc = bias[co];
    for (int c = 0; c < cin; ++c) {
        for (int i = 0; i < kh; ++i) {
            int ih = y * stride + i;
            if (ih < padding || ih >= h + padding) continue;
            int iha = ih - padding;
            for (int j = 0; j < kw; ++j) {
                int iw = x_out * stride + j;
                if (iw < padding || iw >= w + padding) continue;
                int iwa = iw - padding;
                int x_idx = ((nn * cin + c) * h + iha) * w + iwa;
                int w_idx = ((co * cin + c) * kh + i) * kw + j;
                acc = fmaf(x[x_idx], weight[w_idx], acc);
            }
        }
    }
    out[((nn * cout + co) * oh + y) * ow + x_out] = acc;
}

// grad_bias[co] = sum over (nn, y, x_out) of grad_out  (plain add, CPU order)
extern "C" __global__ void conv2d_backward_bias(
    const float* grad_out, float* grad_bias, const int* p
) {
    int n = p[0], cout = p[4], oh = p[7], ow = p[8];
    int co = blockIdx.x * blockDim.x + threadIdx.x;
    if (co >= cout) return;
    float acc = 0.0f;
    for (int nn = 0; nn < n; ++nn)
        for (int y = 0; y < oh; ++y)
            for (int x_out = 0; x_out < ow; ++x_out)
                acc += grad_out[((nn * cout + co) * oh + y) * ow + x_out];
    grad_bias[co] = acc;
}

// grad_weight[co,c,i,j] = sum over (nn, y, x_out) of grad_out * x  (fmaf, CPU order)
extern "C" __global__ void conv2d_backward_weight(
    const float* x, const float* grad_out, float* grad_w, const int* p
) {
    int n = p[0], cin = p[1], h = p[2], w = p[3];
    int cout = p[4], kh = p[5], kw = p[6];
    int oh = p[7], ow = p[8], stride = p[9], padding = p[10];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cout * cin * kh * kw) return;

    int j = idx % kw;
    int i = (idx / kw) % kh;
    int c = (idx / (kw * kh)) % cin;
    int co = idx / (kw * kh * cin);

    float acc = 0.0f;
    for (int nn = 0; nn < n; ++nn) {
        for (int y = 0; y < oh; ++y) {
            int ih = y * stride + i;
            if (ih < padding || ih >= h + padding) continue;
            int iha = ih - padding;
            for (int x_out = 0; x_out < ow; ++x_out) {
                int iw = x_out * stride + j;
                if (iw < padding || iw >= w + padding) continue;
                int iwa = iw - padding;
                int x_idx = ((nn * cin + c) * h + iha) * w + iwa;
                float g = grad_out[((nn * cout + co) * oh + y) * ow + x_out];
                acc = fmaf(g, x[x_idx], acc);
            }
        }
    }
    grad_w[((co * cin + c) * kh + i) * kw + j] = acc;
}

// grad_input[nn,c,p,q] = sum of grad_out * weight over the (co, y, x_out, i, j)
// whose receptive field lands on (p,q). Loop order co,y,x_out,i,j matches CPU.
extern "C" __global__ void conv2d_backward_input(
    const float* weight, const float* grad_out, float* grad_in, const int* p
) {
    int n = p[0], cin = p[1], h = p[2], w = p[3];
    int cout = p[4], kh = p[5], kw = p[6];
    int oh = p[7], ow = p[8], stride = p[9], padding = p[10];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * cin * h * w) return;

    int q = idx % w;
    int pp = (idx / w) % h;
    int c = (idx / (w * h)) % cin;
    int nn = idx / (w * h * cin);

    float acc = 0.0f;
    for (int co = 0; co < cout; ++co) {
        for (int y = 0; y < oh; ++y) {
            for (int x_out = 0; x_out < ow; ++x_out) {
                for (int i = 0; i < kh; ++i) {
                    int ih = y * stride + i;
                    if (ih < padding || ih >= h + padding) continue;
                    if (ih - padding != pp) continue;
                    for (int j = 0; j < kw; ++j) {
                        int iw = x_out * stride + j;
                        if (iw < padding || iw >= w + padding) continue;
                        if (iw - padding != q) continue;
                        int w_idx = ((co * cin + c) * kh + i) * kw + j;
                        float g = grad_out[((nn * cout + co) * oh + y) * ow + x_out];
                        acc = fmaf(g, weight[w_idx], acc);
                    }
                }
            }
        }
    }
    grad_in[idx] = acc;
}

// Max pooling (NCHW). p: [n, c, h, w, k, oh, ow, stride, padding].
// Padded positions treated as -inf (never selected).
extern "C" __global__ void maxpool2d_forward(
    const float* x, float* out, const int* p
) {
    int n = p[0], c = p[1], h = p[2], w = p[3], k = p[4];
    int oh = p[5], ow = p[6], stride = p[7], padding = p[8];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * c * oh * ow) return;

    int x_out = idx % ow;
    int y = (idx / ow) % oh;
    int cc = (idx / (ow * oh)) % c;
    int nn = idx / (ow * oh * c);

    float best = -INFINITY;
    for (int i = 0; i < k; ++i) {
        int ih = y * stride + i;
        if (ih < padding || ih >= h + padding) continue;
        int iha = ih - padding;
        for (int j = 0; j < k; ++j) {
            int iw = x_out * stride + j;
            if (iw < padding || iw >= w + padding) continue;
            int iwa = iw - padding;
            float v = x[((nn * c + cc) * h + iha) * w + iwa];
            if (v > best) best = v;
        }
    }
    out[((nn * c + cc) * oh + y) * ow + x_out] = best;
}

// Backward: grad_in[nn,cc,pp,qq] = sum of grad_out over windows whose argmax is
// (pp,qq). Recomputes each covering window's argmax (first-max, matching forward)
// and iterates windows in (y, x_out) order with plain += to match the CPU order.
extern "C" __global__ void maxpool2d_backward(
    const float* x, const float* grad_out, float* grad_in, const int* p
) {
    int n = p[0], c = p[1], h = p[2], w = p[3], k = p[4];
    int oh = p[5], ow = p[6], stride = p[7], padding = p[8];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * c * h * w) return;

    int qq = idx % w;
    int pp = (idx / w) % h;
    int cc = (idx / (w * h)) % c;
    int nn = idx / (w * h * c);

    float acc = 0.0f;
    for (int y = 0; y < oh; ++y) {
        int i = pp + padding - y * stride;
        if (i < 0 || i >= k) continue;
        for (int x_out = 0; x_out < ow; ++x_out) {
            int j = qq + padding - x_out * stride;
            if (j < 0 || j >= k) continue;
            float best = -INFINITY;
            int bih = -1, biw = -1;
            for (int a = 0; a < k; ++a) {
                int ih = y * stride + a;
                if (ih < padding || ih >= h + padding) continue;
                int iha = ih - padding;
                for (int b = 0; b < k; ++b) {
                    int iw = x_out * stride + b;
                    if (iw < padding || iw >= w + padding) continue;
                    int iwa = iw - padding;
                    float v = x[((nn * c + cc) * h + iha) * w + iwa];
                    if (v > best) { best = v; bih = iha; biw = iwa; }
                }
            }
            if (bih == pp && biw == qq) {
                acc += grad_out[((nn * c + cc) * oh + y) * ow + x_out];
            }
        }
    }
    grad_in[idx] = acc;
}
