#include "main.h"
#include <iostream>
#include "cs_internal.h"
#include "cudalib.h"
#include "elapsed_time.h"
#include <algorithm>
#include <sstream>


static inline int getSign(const float4& v, const float* squad, int i)
{
    i = i * 10;
    const float s0 = squad[i+0];
    const float s1 = squad[i+1];
    const float s2 = squad[i+2];
    const float s3 = squad[i+3];
    const float s4 = squad[i+4];
    const float s5 = squad[i+5];
    const float s6 = squad[i+6];
    const float s7 = squad[i+7];
    const float s8 = squad[i+8];
    const float s9 = squad[i+9];

    const float v0 = v.x;
    const float v1 = v.y;
    const float v2 = v.z;
    const float v3 = v.w;

    const float sum =
            (s0*v0+s1*v1+s2*v2+s3*v3) * v0 +
            (s1*v0+s4*v1+s5*v2+s6*v3) * v1 +
            (s2*v0+s5*v1+s7*v2+s8*v3) * v2 +
            (s3*v0+s6*v1+s8*v2+s9*v3) * v3;


    return (signbit(sum)) ? 0 : 1;
}

std::string cucs_to_coord_string(const float4& spin,
                                 const std::vector<float>& data)
{
    std::stringstream ss;
    for (int i=0;i<data.size()/10;++i)
    {
        bool v = 1==getSign(spin, data.data(), data.size()/10-1-i);
        ss << v?'1':'0';
    }
    return ss.str();
}

void cucs_set_seed(unsigned long long seed)
{
    cu::set_cudarand_seed(seed);
}

std::vector<float4> cucs_compute_random_spinors(size_t count)
{
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cu::VectorType<float4>::type spins = cu::generate_spins(count);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cerr << __FUNCTION__ << ": Time: " << elapsedTime << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::vector<float4> retval(spins.size());
    thrust::copy(spins.begin(), spins.end(), retval.begin());
    return retval;
}

std::vector<float4> cucs_compute_random_unique_spinors(
        const std::vector<float>& spinquads,
        size_t initialCout)
{
    size_t spinquadCount = spinquads.size() / 10;
    cudaEvent_t start, after_gen, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&after_gen);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    thrust::device_vector<float4> spins = cu::generate_spins(initialCout);
    cudaEventRecord(after_gen);
    cu::make_unique_spins(spins, spinquads, spinquadCount);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cerr << __FUNCTION__ << ": Time: " << elapsedTime << " ms\n";
    cudaEventElapsedTime(&elapsedTime, start, after_gen);
    std::cerr << __FUNCTION__ << ": Time<1>: " << elapsedTime << " ms\n";
    cudaEventElapsedTime(&elapsedTime, after_gen, stop);
    std::cerr << __FUNCTION__ << ": Time<2>: " << elapsedTime << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(after_gen);
    cudaEventDestroy(stop);

    std::vector<float4> retval(spins.size());
    thrust::copy(spins.begin(), spins.end(), retval.begin());
    return retval;
}

void cucs_compute_unique_spinors(
        const std::vector<float>& spinquads,
        std::vector<float4>& spinors)
{
    size_t spinquadCount = spinquads.size() / 10;
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    thrust::device_vector<float4> spins = spinors;
    cu::make_unique_spins(spins, spinquads, spinquadCount);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cerr << __FUNCTION__ << ": Time: " << elapsedTime << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    spinors.resize(spins.size());
    thrust::copy(spins.begin(), spins.end(), spinors.begin());
}

void cucs_compute_spinors_and_neighbours(
        const std::vector<float>& spinquads,
        size_t initialCout,
        /*out*/ std::vector<float4>& spinors,
        /*out*/ std::vector<u_int32_t>& neighbour_index_1,
        /*out*/ std::vector<u_int32_t>& neighbour_index_2)
{
    size_t spinquadCount = spinquads.size() / 10;
    cudaEvent_t start, after_gen, after_uniq, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&after_gen);
    cudaEventCreate(&after_uniq);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    thrust::device_vector<float4> spins = cu::generate_spins(initialCout);
    cudaEventRecord(after_gen);
    cu::make_unique_spins(spins, spinquads, spinquadCount);
    cudaEventRecord(after_uniq);
    cu::locate_pairs(spins, spinquads, spinquadCount, neighbour_index_1, neighbour_index_2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cerr << __FUNCTION__ << ": Time: " << elapsedTime << " ms\n";
    cudaEventElapsedTime(&elapsedTime, start, after_gen);
    std::cerr << __FUNCTION__ << ": Time<1>: " << elapsedTime << " ms\n";
    cudaEventElapsedTime(&elapsedTime, after_gen, after_uniq);
    std::cerr << __FUNCTION__ << ": Time<2>: " << elapsedTime << " ms\n";
    cudaEventElapsedTime(&elapsedTime, after_uniq, stop);
    std::cerr << __FUNCTION__ << ": Time<3>: " << elapsedTime << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(after_gen);
    cudaEventDestroy(after_uniq);
    cudaEventDestroy(stop);

    spinors.resize(spins.size());
    thrust::copy(spins.begin(), spins.end(), spinors.begin());
}

void cucs_compute_neighbours(
        const std::vector<float>& spinquads,
        const std::vector<float4>& spinors,
        /*out*/ std::vector<u_int32_t>& neighbour_index_1,
        /*out*/ std::vector<u_int32_t>& neighbour_index_2)
{
    size_t spinquadCount = spinquads.size() / 10;
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    thrust::device_vector<float4> spins = spinors;
    cu::locate_pairs(spins, spinquads, spinquadCount, neighbour_index_1, neighbour_index_2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cerr << __FUNCTION__ << ": Time: " << elapsedTime << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
