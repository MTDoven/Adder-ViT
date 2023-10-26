import torch

_help_words = \
""" 
reference: https://blog.csdn.net/qq_41910905/article/details/109650182
Use this pycuda library to accelarate your codes
You may meet some strange problem when install this library. Just be patient.
"""
try:
    import pycuda.autoprimaryctx
    from pycuda.compiler import SourceModule
except:
    raise ImportError("Cannot import pycuda.\n"+_help_words)
    
torch.cuda.FloatTensor(8) # This is a useless code for cuda_init. It will lead to a bug without it.



kernel_code = r"""

__global__ void ADD(const float *X, const float *W, const int *width, float *O)
{
    const int line_idx = blockIdx.x;
    const int colm_idx = threadIdx.x;
    const int out_dim = blockDim.x;
    const int in_dim = width[0];

    float value=0.0;
    float* x = (float*)X + line_idx * in_dim;
    float* w = (float*)W + colm_idx;
    const float* stop_sign = x + in_dim;
    for(; x<stop_sign; x++, w+=out_dim){
        value += fabs((*x) - (*w)); }
    
    O[line_idx * out_dim + colm_idx] = -value;
} //checked

__global__ void ADD_BACKWARD_X(const float *X, const float *W, const float *grad_in, const int *out, float *O)
{
    const int line_idx = blockIdx.x;
    const int colm_idx = threadIdx.x;
    const int out_dim = out[0];
    const int mid_dim = blockDim.x;

    float value = 0.0;
    float* w = (float*)W + colm_idx*out_dim;
    float* g = (float*)grad_in + line_idx*out_dim;
    const float* stop_sign = w + out_dim;
    const float value_x = X[line_idx * mid_dim + colm_idx];
    for(; w<stop_sign; w++, g++){
        float temp = (*w) - value_x;
        temp = max(-1.0, min(4*temp, 1.0));
        value += ( temp * (*g) ); }
    
    O[line_idx * mid_dim + colm_idx] = value;
}

__global__ void ADD_BACKWARD_W(const float *X, const float *W, const float *grad_in, const int *length, float *O)
{
    const int line_idx = blockIdx.x;
    const int colm_idx = threadIdx.x;
    const int out_dim = blockDim.x;
    const int mid_dim = gridDim.x;
    const int in_dim = length[0];

    float value = 0.0;
    float* x = (float*)X + line_idx;
    float* g = (float*)grad_in + colm_idx;
    const float* stop_sign = x + mid_dim * in_dim;
    const float value_w = W[line_idx * out_dim + colm_idx];
    for(; x<stop_sign; x+=mid_dim, g+=out_dim){
        float temp = (*x) - value_w;
        temp = max(-2.0, min(temp, 2.0));
        value += ( temp * (*g) ); }

    O[line_idx * out_dim + colm_idx] = value;
}



__global__ void MH_ADD(const float *Q, const float *K, const int *shapes, float *O)
{
    const int x_now = blockIdx.x;
    const int y_now = blockIdx.y;
    const int z_now = threadIdx.x;
    const int q_per_line = shapes[2];
    const int q_per_piece = shapes[1]*shapes[2];
    const int k_per_line = shapes[5];
    const int k_per_piece = shapes[4]*shapes[5];

    float value=0.0;
    float* q = (float*)Q + x_now*q_per_piece + y_now*q_per_line;
    float* k = (float*)K + x_now*k_per_piece + z_now;
    const float* stop_sign = q + q_per_line;
    for(; q<stop_sign; q++, k+=k_per_line){
        value += fabs((*q)-(*k)); }

    O[x_now*gridDim.y*blockDim.x + y_now*blockDim.x + z_now] = -value;
}

__global__ void MH_ADD_BACKWARD_Q(const float *Q, const float *K, const float *grad, const int *shapes, float *O)
{
    const int x_now = blockIdx.x;
    const int y_now = blockIdx.y;
    const int z_now = threadIdx.x;
    const int q_per_line = shapes[2];
    const int q_per_piece = shapes[1]*shapes[2];
    const int k_per_line = shapes[5];
    const int k_per_piece = shapes[4]*shapes[5];
    const int a_per_piece = shapes[1]*shapes[5];
    const int a_per_line = k_per_line;

    float grad_q=0.0;
    const float q_value = Q[x_now*q_per_piece + y_now*q_per_line + z_now];
    float* k = (float*)K + x_now*k_per_piece + z_now*k_per_line;
    float* g = (float*)grad + x_now*a_per_piece + y_now*a_per_line;
    const float* stop_sign = k + k_per_line;
    for(; k<stop_sign; k++, g++){
        float temp = (*k) - q_value;
        temp = max(-1.0, min(4*temp, 1.0));
        grad_q += temp * (*g); }

    O[x_now*q_per_piece + y_now*q_per_line + z_now] = grad_q;
}

__global__ void MH_ADD_BACKWARD_K(const float *Q, const float *K, const float *grad, const int *shapes, float *O)
{
    const int x_now = blockIdx.x;
    const int y_now = blockIdx.y;
    const int z_now = threadIdx.x;
    const int q_per_colm = shapes[1];
    const int q_per_line = shapes[2];
    const int q_per_piece = shapes[1]*shapes[2];
    const int k_per_line = shapes[5];
    const int k_per_piece = shapes[4]*shapes[5];
    const int a_per_piece = shapes[1]*shapes[5];
    const int a_per_line = k_per_line;

    float grad_k=0.0;
    float k_value = K[x_now*k_per_piece + y_now*k_per_line + z_now];
    float* q = (float*)Q + x_now*q_per_piece + y_now;
    float* g = (float*)grad + x_now*a_per_piece + z_now;
    float* stop_sign = q + q_per_colm*q_per_line;
    for(; q<stop_sign; q+=q_per_line, g+=a_per_line) {
        float temp = (*q) - k_value;
        temp = max(-1.0, min(4*temp, 1.0));
        grad_k += temp * (*g); }
    
    O[x_now*k_per_piece + y_now*k_per_line + z_now] = grad_k;
}


"""
mod = SourceModule(kernel_code)

# To transfer the torch.tensor to pycuda's pointer
class Holder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()
    def get_pointer(self):
        return self.t.data_ptr()


MH_ADD = mod.get_function("MH_ADD")
MH_ADD_BACKWARD_Q = mod.get_function("MH_ADD_BACKWARD_Q")
MH_ADD_BACKWARD_K = mod.get_function("MH_ADD_BACKWARD_K")
ADD = mod.get_function("ADD")
ADD_BACKWARD_X = mod.get_function("ADD_BACKWARD_X")
ADD_BACKWARD_W = mod.get_function("ADD_BACKWARD_W")
