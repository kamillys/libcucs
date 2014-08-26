#include "main.h"
#include <iostream>
#include "cs_internal.h"
#include "cudalib.h"
#include "elapsed_time.h"
#include <algorithm>
#include <sstream>

template<typename T>
T compute_hash_T(const std::vector<bool>& v)
{
    T h(0);
    T sh(0);
    for (size_t i = 0 ; i <  v.size(); ++i)
    {
        T val = v[i] ? 1 : 0;
        h ^= val << sh;
        sh = (sh+1) % (8*sizeof(T));
    }
    return h;
}

u_int32_t compute_hash_32(const std::vector<bool>& v)
{
    u_int32_t h (0);
    u_int32_t sh = 0;
    for (size_t i = 0 ; i <  v.size(); ++i)
    {
        u_int32_t val = v[i] ? 1 : 0;
        h ^= val << sh;
        sh = (sh+1) % 32;
    }
    return h;
}

template<typename T>
bool isPowerOfTwo (T x)
{
    return ((x != 0) && ((x & (~x + 1)) == x));
}

u_int32_t NumberOfSetBits(u_int32_t i)
{
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

u_int64_t NumberOfSetBits(u_int64_t i)
{
    i = i - ((i >> 1) & 0x5555555555555555UL);
    i = (i & 0x3333333333333333UL) + ((i >> 2) & 0x3333333333333333UL);
    return (((i + (i >> 4)) & 0xF0F0F0F0F0F0F0FUL) * 0x101010101010101UL) >> 56;
}

inline int getSign(const float4& v, const float* squad, int i = 0)
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

static inline std::string toCoordString(const float4& spin,
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

/************************************************************************
 ************************************************************************
 ************************************************************************
 ************************************************************************
 ***********************************************************************/

void fill_zeros(std::vector<float>& data)
{
    size_t HB = HASHBITS * 10;
    size_t size = data.size();
    size_t toAdd = HB - (size%HB);
    if(toAdd != HB)
        data.resize(size+toAdd, 0);
}

void foobar()
{
    /*
        glm::mat4 d1 = glm::make_mat4(data[i].m);
        bool v = glm::dot(s1,  (d1 * s1)) >= 0;
        q[i] = v;
     */
    std::vector<float> data = cu::readFile("/home/kamil/Projekty/Mgr/cucs_demo/build/spin_quadrics_compress.txt");
    size_t spinquadCount = data.size() / 10;
    //thrust::device_vector<float> data = h_data;
    //Resize
    //fill_zeros(data);

    size_t count = 1000000;
    //size_t count = 256;

    double startTime = getCPUTime();

    cudaEvent_t start, stop, after_gen, after_uniq;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&after_gen);
    cudaEventCreate(&after_uniq);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cu::VectorType<float4>::type d_spins;
    d_spins = cu::generate_spins(count);
    cudaEventRecord(after_gen);
    cu::make_unique_spins(d_spins, data, spinquadCount);
    cudaEventRecord(after_uniq);
    cu::locate_pairs(d_spins, data, spinquadCount);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    double endTime = getCPUTime();
    std::cout << "Time: " << elapsedTime << " ms\n";
    std::cout << "Time <alt>: " << (endTime - startTime)*1000 << "ms \n";
    cudaEventElapsedTime(&elapsedTime, start, after_gen);
    std::cout << "Time <1>: " << elapsedTime << " ms\n";
    cudaEventElapsedTime(&elapsedTime, after_gen, after_uniq);
    std::cout << "Time <2>: " << elapsedTime << " ms\n";
    cudaEventElapsedTime(&elapsedTime, after_uniq, stop);
    std::cout << "Time <3>: " << elapsedTime << " ms\n";

    thrust::host_vector<float4> spins = d_spins;

    for (int j=0;j<10/*spins.size()*/;++j)
    //for (int j=0;j<spins.size();++j)
    {
        std::cout << toCoordString(spins[j], data) << std::endl;
    }
    //for(int i=0;i<10;++i)
    //    std::cout << d_hashes[i] << "\n";
}

void cucs_entry()
{
    unsigned long long seed = 2568305073;// time(NULL);
    //unsigned long long seed = time(NULL);
    cu::set_cudarand_seed(seed);
    foobar();
    foobar();
}
