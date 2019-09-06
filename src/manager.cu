#include <assert.h>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include <kernel.cu>
#include <manager.hh>

using namespace std;

      
Heuns::Heuns (
    int num_oscilators_, int num_couplings_,
    int num_indices_, int num_ptr_,
    int block_size_,
    float *omegas_host, float *phases_host, 
    float *couplings_host,
    int* indices_host, int* ptr_host
  ) {

  num_oscilators = num_oscilators_;
  num_couplings = num_couplings_;
  num_indices = num_indices_;
  num_ptr = num_ptr_;
  block_size = block_size_;


  num_oscilators_float = (float)num_oscilators;
  num_oscilators_all = num_oscilators*num_couplings;
  num_blocks = (num_oscilators_all + block_size - 1) / block_size;
    
  cudaMalloc(&indices_d, num_indices*sizeof(int));
  cudaMalloc(&ptr_d, num_ptr*sizeof(int));
      
 	cudaMemcpy(indices_d, indices_host, num_indices*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(ptr_d, ptr_host, num_ptr*sizeof(int), cudaMemcpyHostToDevice);
    
  cudaMalloc(&omegas_d, num_oscilators*sizeof(float));
 	cudaMemcpy(omegas_d, omegas_host, num_oscilators*sizeof(float), cudaMemcpyHostToDevice);
    
  cudaMalloc(&phases_old_d, num_oscilators_all*sizeof(float));
  cudaMalloc(&phases_new_d, num_oscilators_all*sizeof(float));
      
	cudaMemcpy(phases_old_d, phases_host, num_oscilators_all*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(phases_new_d, phases_host, num_oscilators_all*sizeof(float), cudaMemcpyHostToDevice);
 
  cudaMalloc(&couplings_d, num_couplings*sizeof(float));
	cudaMemcpy(couplings_d, couplings_host, num_couplings*sizeof(float), cudaMemcpyHostToDevice);
    
}
    
void Heuns::heuns_step(float dt, bool reverse){
  if (reverse){
    kernel_heuns<<<num_blocks, block_size>>>(num_oscilators, num_couplings,
      phases_old_d, phases_new_d, omegas_d, couplings_d, dt, indices_d, ptr_d);
  }else{
    kernel_heuns<<<num_blocks, block_size>>>(num_oscilators, num_couplings,
      phases_new_d, phases_old_d, omegas_d, couplings_d, dt, indices_d, ptr_d);  
  }
  cudaDeviceSynchronize();
}
    
void Heuns::heuns(int num_temps, float dt){
  bool reverse;
  for (int i_temp = 0; i_temp < num_temps; i_temp++){
    reverse = i_temp % 2 != 0;
    this->heuns_step(dt, reverse);
  }
}    
    
void Heuns::calc_order_parameter(int num_temps, float dt, float *order_parameter_list){
    
    thrust::device_ptr<float> d_phases_new(phases_new_d); 
    thrust::device_ptr<float> d_phases_old(phases_old_d); 

    thrust::device_vector<int> d_keys_couplings(num_oscilators_all);
    thrust::host_vector<int> h_keys_output(num_couplings);
   
    thrust::device_vector<float> d_reduce_result_sin(num_couplings);
    thrust::device_vector<float> d_reduce_result_cos(num_couplings);
    for (int i = 0; i<num_couplings; i++){
        thrust::fill(
            d_keys_couplings.begin() + i*num_oscilators, 
            d_keys_couplings.begin()+ (i+1)*num_oscilators, 
            i
        );
        h_keys_output[i]=i;
    }
    thrust::device_vector<int> d_keys_output =h_keys_output;

    CosFunctor<float> cos_op;
    SinFunctor<float> sin_op;
    
    bool reverse;
    for (int i_temp = 0; i_temp <num_temps; i_temp++){
        reverse = i_temp % 2 != 0;
        this->heuns_step(dt, reverse);
        // calc order parameter
        if (reverse){
      
          thrust::reduce_by_key(
            d_keys_couplings.begin(), d_keys_couplings.end(),
            thrust::make_transform_iterator(d_phases_old, sin_op),
            // thrust::make_discard_iterator(),
             d_keys_output.begin(),
            d_reduce_result_sin.begin()
          );
          thrust::reduce_by_key(
            d_keys_couplings.begin(), d_keys_couplings.end(),
            thrust::make_transform_iterator(d_phases_old, cos_op),
            //thrust::make_discard_iterator(),
             d_keys_output.begin(),
            d_reduce_result_cos.begin()
          );

        }else{

          thrust::reduce_by_key(
            d_keys_couplings.begin(), d_keys_couplings.end(),
            thrust::make_transform_iterator(d_phases_new, sin_op),
            thrust::make_discard_iterator(),
            d_reduce_result_sin.begin()
          );
          thrust::reduce_by_key(
            d_keys_couplings.begin(), d_keys_couplings.end(),
            thrust::make_transform_iterator(d_phases_new, cos_op),
            thrust::make_discard_iterator(),
            d_reduce_result_cos.begin()
          );

        }

        thrust::transform(
          d_reduce_result_sin.begin(),
          d_reduce_result_sin.end(),
          d_reduce_result_cos.begin(),
          d_reduce_result_cos.begin(),
          OrderFunctor(num_oscilators_float)
        );
        
        for(int i_coupling=0; i_coupling<num_couplings; i_coupling++){
            order_parameter_list[i_coupling*num_temps + i_temp] = d_reduce_result_cos[i_coupling];
        }
    }
}

void Heuns::get_phases (float* phases_host) {

  int size = num_oscilators_all * sizeof(float);
  cudaMemcpy(phases_host, phases_new_d, size, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  assert(err == 0);
    
}

Heuns::~Heuns() {
    
  cudaFree(indices_d);
  cudaFree(ptr_d);

  cudaFree(omegas_d);

  cudaFree(phases_old_d);
  cudaFree(phases_new_d);

  cudaFree(couplings_d);
}