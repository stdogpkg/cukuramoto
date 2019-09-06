#include <stdio.h>


template <typename T>
struct SinFunctor
{
    __host__ __device__
    T operator()(const T& x) const {
        return sinf(x);
    }
};


template <typename T>
struct CosFunctor
{
    __host__ __device__
    T operator()(const T& x) const {
        return cosf(x);
    }
};


struct OrderFunctor
{
    const float num_oscilators;

    OrderFunctor(float _num_oscilators) : num_oscilators(_num_oscilators) {}
    __host__ __device__
        float operator()(const float& x, const float& y) const {
            return sqrtf(powf(x, 2)+powf(y, 2))/num_oscilators;
        }
};


__global__ void kernel_heuns(int num_oscilators, int num_couplings, float *phases_new,
  float *phases_old, float *omegas,float *couplings, float dt,  int *indices, int *ptr)
{

    int i_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int index =i_thread; index < num_oscilators*num_couplings; index += stride){

        float findex = (float)index;
        float fnum_oscilators = (float)num_oscilators;
        float fi_section = (findex+1)/fnum_oscilators-1;
        fi_section=ceil(fi_section);
        int i_section = (int)fi_section;
        
        //int i_section = (int)ceil(((float)index+1)/(float)num_oscilators-1);
        int i_node = index - i_section*num_oscilators;

        float coupling = couplings[i_section];
        float phase = phases_old[index];
        float omega = omegas[i_node];

        int i_nei = 0;
        float f = 0;
        float f_tild = 0;
        float phase_tild = 0.;
        for( int i_ptr = ptr[i_node]; i_ptr < ptr[i_node+1]; i_ptr = i_ptr + 1 ) {
            i_nei = indices[i_ptr]+ i_section*num_oscilators;
            f +=  coupling*sinf( phases_old[i_nei]  -phase);

        };
        f = f + omega;

        phase_tild =  phase + dt*f;

        for( int i_ptr = ptr[i_node]; i_ptr < ptr[i_node+1]; i_ptr = i_ptr + 1 ) {
            i_nei = indices[i_ptr]+ i_section*num_oscilators;
            f_tild +=  coupling*sinf( phases_old[i_nei] - phase_tild);
        };
        f_tild = f_tild +  omega;
        phases_new[index] = phase + dt*(f+f_tild)/2.;
    }
}