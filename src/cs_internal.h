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
                       const std::vector<float>& spinquadrics,
                       size_t spinquadCount);

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

void compute_hash_part(thrust::device_vector<float4>& spins,
                       const std::vector<float>& spinquadrics,
                       thrust::device_vector<hashtype>& hashPart,
                       int i,
                       size_t& rem);

void locate_pairs(thrust::device_vector<float4>& spins,
                  const std::vector<float>& spinquadrics,
                  size_t spinquadCount, std::vector<u_int32_t>& output1, std::vector<u_int32_t>& output2);


}
