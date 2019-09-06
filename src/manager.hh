#include <stdio.h>

class Heuns {

  int num_oscilators, num_couplings;
  int num_indices, num_ptr;
  int block_size;
  int num_oscilators_float;
  int num_oscilators_all;
  int num_blocks;
   
  int *indices_d, *ptr_d;
      
  float *omegas_d;
    
  float *phases_old_d;
  float *phases_new_d;
   
  float *couplings_d; 

public:

  Heuns(
      int num_oscilators, int num_couplings,
      int num_indices, int num_ptr,
      int block_size,
      float *omegas, float *phases, float *couplings,
      int *indices, int *ptr
  );

  ~Heuns(); 

  void get_phases (float* phases_host);        
          
  void heuns_step(float dt, bool reverse);
  void heuns(int num_temps, float dt);
  void heuns_after_transient(int num_temps, float dt);
  void calc_order_parameter(int num_temps, float dt, float *order_parameter_list);

};