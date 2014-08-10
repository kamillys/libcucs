#include <iostream>

#include "cs_internal.h"
#include "cudalib.h"

#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <bitset>


namespace cu
{

__constant__ static float squad[10*HASHBITS];
static texture<float4, 1, cudaReadModeElementType> spinTex;
static texture<hashtype, 1, cudaReadModeElementType> coordPartTex;

__device__
inline static int getSign(float v0, float v1, float v2, float v3, int i)
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

    const float sum =
            (s0*v0+s1*v1+s2*v2+s3*v3) * v0 +
            (s1*v0+s4*v1+s5*v2+s6*v3) * v1 +
            (s2*v0+s5*v1+s7*v2+s8*v3) * v2 +
            (s3*v0+s6*v1+s8*v2+s9*v3) * v3;


    return (signbit(sum)) ? 0 : 1;
    //return signbit(sum);
}

struct ComputeHash
{
    u_int32_t count;

    ComputeHash(size_t s)
        : count(s) { }

    __device__
    hashtype operator()(const u_int32_t& index)
    {
        const float v0=tex1Dfetch(spinTex, index).x;
        const float v1=tex1Dfetch(spinTex, index).y;
        const float v2=tex1Dfetch(spinTex, index).z;
        const float v3=tex1Dfetch(spinTex, index).w;

        hashtype retval(0); int offset = 0;
        //for (size_t i=0;i<count;++i)
        for (u_int32_t i=0;i<count;++i)
        {
            retval ^= getSign(v0,v1,v2,v3, i) << offset;
            offset = (offset+1)& HASH_MASK;
        }
        return retval;
    }
};

struct SpinSortLt
{
    size_t count;

    SpinSortLt(size_t s)
        : count(s) { }

    __device__
    //__host__
    bool operator() (const float4& a, const float4& b)
    {
        // a < b
        //return getSign(a, d1) < getSign(b, d1);
        int j=0;
        //for (int i = count - 1; i >= 0; i--)
        for (int i = 0; i < count; i++)
        {
            j++;
            if (j > 3) break;
            //int s1 = getSign(a, i);
            //int s2 = getSign(b, i);
            //if (s1 == s2) continue;
            //return s1 < s2;
        }
        return false;
    }
};

struct SpinorsSorter
{
    __device__
    bool operator()(const u_int32_t& _a, const u_int32_t& _b)
    {
        return tex1Dfetch(coordPartTex, _a) < tex1Dfetch(coordPartTex, _b);
    }
};

struct ReorderCoords
{
    __device__
    hashtype operator()(const u_int32_t& _a)
    {
        return tex1Dfetch(coordPartTex, _a);
    }
};

struct ReorderSpins
{
    __device__
    float4 operator()(const u_int32_t& _a)
    {
        return tex1Dfetch(spinTex, _a);
    }
};

struct SpinHashChecker
{
    __device__
    u_int32_t operator()(const u_int32_t& index, const u_int32_t& value)
    {
        if (index == 0)
            return 0;
        u_int32_t v0 = tex1Dfetch(coordPartTex, index-1);
        u_int32_t v1 = tex1Dfetch(coordPartTex, index);
        if (v0 == v1)
            return value+1;
        return value;
    }
};

struct SpinIndexCleaner
{
    u_int32_t count;

    SpinIndexCleaner(size_t s)
        : count(s) { }

    __device__
    bool operator()(const u_int32_t& index)
    {
        u_int32_t value = tex1Dfetch(coordPartTex, index);
        return value == count;
    }
};

void compute_hash_part(thrust::device_vector<float4>& spins,
                       std::vector<float>& spinquadrics,
                       thrust::device_vector<hashtype>& hashPart,
                       int i,
                       size_t& rem)
{
    size_t toCpy = std::min<size_t>(rem, HASHBITS);

    cutilSafeCall( cudaBindTexture(NULL, spinTex, thrust::raw_pointer_cast(spins.data()), sizeof(float4)*spins.size()) );
    cutilSafeCall( cudaMemcpyToSymbol(squad, spinquadrics.data()+i*10*HASHBITS, sizeof(float)*10*toCpy) );
    thrust::transform(thrust::counting_iterator<u_int32_t>(0),
                      thrust::counting_iterator<u_int32_t>(spins.size()),
                      hashPart.begin(), ComputeHash(toCpy));
    cutilSafeCall( cudaUnbindTexture(spinTex) );
}

void make_unique_spins(thrust::device_vector<float4>& spins,
                       std::vector<float>& spinquadrics,
                       size_t spinquadCount)
{
    size_t parts = inc_div<size_t>(spinquadrics.size()/10, HASHBITS);
    std::vector<thrust::device_vector<hashtype> > hashPart(parts);

    /*
     * TODO:
     * Refactor code.
     * - Sort & compute parts
     * - Simpler reorder
     * - Reduce & compute parts
    */
    {
        std::vector<size_t> partsSizes(parts);
        size_t rem = spinquadCount;
        for (int i=0;i<parts; ++i)
        {
            partsSizes[i] = std::min<size_t>(rem, HASHBITS);
            rem -= HASHBITS;
            hashPart[i].resize(spins.size());
            compute_hash_part(spins, spinquadrics, hashPart[i], i, partsSizes[i]);
        }
//        for (int j=0;j<10;++j)
//        {
//            for (int i=0;i<parts; ++i)
//                std::cerr << std::bitset<32>(hashPart[parts-1-i][j]);
//            std::cerr << std::endl;
//        }
    }

    thrust::device_vector<u_int32_t> elemsIds(spins.size());
    thrust::sequence(elemsIds.begin(), elemsIds.end());
//    for (int i=0;i<10;++i)
//    {
//        std::cerr << elemsIds[i] << " => ";
//        for (int j=0;j<parts; ++j)
//            std::cerr << std::bitset<32>(hashPart[parts-1-j][elemsIds[i]]);
//        std::cerr << std::endl;
//    }
//    std::cerr << "======================================" << std::endl;
    for (int i=0;i<parts; ++i)
    {
        cutilSafeCall( cudaBindTexture(NULL, coordPartTex, thrust::raw_pointer_cast(hashPart[i].data()), sizeof(hashtype)*hashPart[i].size()) );
        thrust::stable_sort(elemsIds.begin(), elemsIds.end(), SpinorsSorter());
        cutilSafeCall( cudaUnbindTexture(coordPartTex) );
//        for (int i=0;i<10;++i)
//        {
//            std::cerr << elemsIds[i] << " => ";
//            for (int j=0;j<parts; ++j)
//                std::cerr << std::bitset<32>(hashPart[parts-1-j][elemsIds[i]]);
//            std::cerr << std::endl;
//        }
//        std::cerr << "======================================" << std::endl;
    }
    //Reorder
    {
        thrust::device_vector<hashtype> hash_tmp(spins.size());
        for (int i=0;i<parts; ++i)
        {
            cutilSafeCall( cudaBindTexture(NULL, coordPartTex, thrust::raw_pointer_cast(hashPart[i].data()), sizeof(hashtype)*hashPart[i].size()) );
            thrust::transform(elemsIds.begin(), elemsIds.end(), hash_tmp.begin(), ReorderCoords());
            cutilSafeCall( cudaUnbindTexture(coordPartTex) );
            hashPart[i].swap(hash_tmp);
        }
        hash_tmp.clear();
        thrust::device_vector<float4> spins_tmp(spins.size());
        cutilSafeCall( cudaBindTexture(NULL, spinTex, thrust::raw_pointer_cast(spins.data()), sizeof(float4)*spins.size()) );
        thrust::transform(elemsIds.begin(), elemsIds.end(), spins_tmp.begin(), ReorderSpins());
        cutilSafeCall( cudaUnbindTexture(spinTex) );
        std::swap(spins, spins_tmp);
        spins_tmp.clear();
    }
    //Make unique
    //Algo: Check if elem before has the same hash
    //if yes, +1
    //Remove all that has counter equal to part count
    {

        thrust::device_vector<u_int32_t> hash_counter(spins.size());
        thrust::fill(hash_counter.begin(), hash_counter.end(), 0);
        thrust::sequence(elemsIds.begin(), elemsIds.end());
        for (int i=0;i<parts; ++i)
        {
            cutilSafeCall( cudaBindTexture(NULL, coordPartTex, thrust::raw_pointer_cast(hashPart[i].data()), sizeof(hashtype)*hashPart[i].size()) );
            thrust::transform(elemsIds.begin(), elemsIds.end(), hash_counter.begin(), hash_counter.begin(), SpinHashChecker());
            cutilSafeCall( cudaUnbindTexture(coordPartTex) );
        }
        cutilSafeCall( cudaBindTexture(NULL, coordPartTex, thrust::raw_pointer_cast(hash_counter.data()), sizeof(hashtype)*hash_counter.size()) );
        elemsIds.resize(thrust::remove_if(elemsIds.begin(), elemsIds.end(), SpinIndexCleaner(parts)) - elemsIds.begin());
        cutilSafeCall( cudaUnbindTexture(coordPartTex) );

        //std::cerr << "Remaining: " << elemsIds.size() << std::endl;
        //Final Reorder
        thrust::device_vector<float4> final_spins(elemsIds.size());
        cutilSafeCall( cudaBindTexture(NULL, spinTex, thrust::raw_pointer_cast(spins.data()), sizeof(float4)*spins.size()) );
        thrust::transform(elemsIds.begin(), elemsIds.end(), final_spins.begin(), ReorderSpins());
        cutilSafeCall( cudaUnbindTexture(spinTex) );
        spins.swap(final_spins);
    }

}

}
