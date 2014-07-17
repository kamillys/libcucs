#pragma once

#include <vector>
#include <string>

#include "cudaptr.h"

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

namespace cu
{
template<typename T>
struct VectorType { typedef thrust::device_vector<T> type; };

std::vector<float> readFile(std::string path);

void set_cudarand_seed(unsigned long long seed);
VectorType<float4>::type generate_spins(size_t count);

void make_unique_spins(thrust::device_vector<float4>& spins,
                       std::vector<float>& spinquadrics, size_t spinquadCount);

#if 0 || USE_64BIT_HASH
typedef u_int64_t hashtype;
#define HASH_MASK 0x3F
#define HASHSIZE 8
#else
typedef u_int32_t hashtype;
#define HASH_MASK 0x1F
#define HASHSIZE 4
#endif
#define HASHBITS (8*HASHSIZE)


//void compute_hashes(thrust::device_vector<hashtype>& hashes,
//                    thrust::device_vector<glm::vec4>& spins,
//                    thrust::device_vector<glm::mat4>& spinquadrics);

//void make_hashmap(thrust::device_vector<hashtype>& d_hashmap,
//                  thrust::device_vector<hashtype>& d_hashes);

}
