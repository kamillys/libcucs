#include <iostream>
#include <curand.h>
#include "cs_internal.h"
#include "cudalib.h"
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/random.h>

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n"; \
    throw 1;}} while(0)

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

struct is_over_one
{
  __host__ __device__
  bool operator()(const float4& v)
  {
    return (v.x*v.x + v.y*v.y + v.z*v.z) > 1.0f;
  }
};

__device__ __host__
float4 random_spin(const float u1, const float u2, const float u3)
{
    const float ONE    = 1.0f;
    const float TWO    = 2.0f;

    const float TWO_PI = TWO * M_PI_F;

    const float x = sqrtf(ONE - u1) * sinf(TWO_PI * u2);
    const float y = sqrtf(ONE - u1) * cosf(TWO_PI * u2);
    const float z = sqrtf(u1) * sinf(TWO_PI * u3);
    const float w = sqrtf(u1) * cosf(TWO_PI * u3);
    return make_float4(x,y,z,w);
}

struct make_spin
{
    __host__ __device__
    float4 operator()(const float3& v)
    {
        return random_spin(v.x, v.y, v.z);
    }
};

namespace cu {

static curandGenerator_t gen;

__attribute__((constructor))
static void initialize_cuda_rand_generator()
{
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
}

__attribute__((destructor))
static void destroy_cuda_rand_generator()
{
    CURAND_CALL(curandDestroyGenerator(gen));
}

void set_cudarand_seed(unsigned long long seed)
{
    // Set seed
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
}

void generate_sequence(thrust::device_vector<float3>& d_data, size_t count)
{
    d_data.resize(count);
    // Generate 3n floats on device
    CURAND_CALL(curandGenerateUniform(gen, reinterpret_cast<float*>(thrust::raw_pointer_cast(d_data.data())), 3*count));
}

VectorType<float4>::type generate_spins(size_t count)
{
    thrust::device_vector<float4> d_spins;
    d_spins.resize(count);
    thrust::device_vector<float3> d_random_floats;
    cu::generate_sequence(d_random_floats, count);
    thrust::transform(d_random_floats.begin(), d_random_floats.end(), d_spins.begin(), make_spin() );

    d_spins.resize(thrust::remove_if(d_spins.begin(), d_spins.end(), is_over_one()) - d_spins.begin());
    //std::cerr << "Valid: " << d_spins.size() << "\n";
    return d_spins;
}

}
