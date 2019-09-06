from libcpp cimport bool
import numpy as np
cimport numpy as np


cdef extern from "src/manager.hh":
    cdef cppclass Cpp_Heuns "Heuns":
        Cpp_Heuns(
            int, int,
            int, int,
            int,
            np.float32_t*,  np.float32_t*,np.float32_t*,
            np.int32_t*,np.int32_t*
            
        )
        void heuns_step(float, bool)
        
        void heuns_after_transient(int, float)
        void heuns(int, float)
        void get_phases(np.float32_t*)
        void calc_order_parameter(int, float, np.float32_t*)


cdef class Heuns:
    cdef Cpp_Heuns* cpp_heuns
    cdef int dim1 
    cdef int num_oscilators_all
    cdef int num_couplings

    def __cinit__(
            self,
            int num_oscilators,
            int block_size,
            np.ndarray[ndim=1, dtype=np.float32_t] omegas, 
            np.ndarray[ndim=1, dtype=np.float32_t] phases, 
            np.ndarray[ndim=1, dtype=np.float32_t] couplings, 
            np.ndarray[ndim=1, dtype=np.int32_t] indices,
            np.ndarray[ndim=1, dtype=np.int32_t] ptr
        ):
        self.num_oscilators_all = num_oscilators*len(couplings)
        self.num_couplings = len(couplings)
        cdef int num_indices = len(indices)
        cdef int num_ptr = len(ptr)
        self.cpp_heuns = new Cpp_Heuns(
            num_oscilators, self.num_couplings,
            num_indices, num_ptr,
            block_size,
            &omegas[0], 
            &phases[0], 
            &couplings[0], 
            &indices[0],
            &ptr[0]
        )
  
    def get_phases(self):
        cdef np.ndarray[ndim=1, dtype=np.float32_t] phases = np.zeros(
            self.num_oscilators_all, dtype=np.float32)

        self.cpp_heuns.get_phases(&phases[0])

        return phases
    
    def heuns_step(self,float dt, bool reverse=False):
      
        self.cpp_heuns.heuns_step(dt, reverse)

    def heuns(self, int num_temps, float dt):
      
        self.cpp_heuns.heuns(num_temps, dt)
      
    def get_order_parameter(self, int num_temps, float dt):
      
        cdef np.ndarray[ndim=1, dtype=np.float32_t] order_parameter_list = np.zeros(
            self.num_couplings*num_temps, dtype=np.float32)

        self.cpp_heuns.calc_order_parameter(
            num_temps, dt, &order_parameter_list[0])
        
        return order_parameter_list
