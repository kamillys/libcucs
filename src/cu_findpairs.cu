#include "cs_internal.h"
#include "cudalib.h"
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <bitset>

#define CU_INVALID_INDEX_ITEM (static_cast<u_int32_t>(~0))


namespace cu
{

static texture<hashtype, 1, cudaReadModeElementType> cuFpCoordsTex;
static texture<u_int32_t, 1, cudaReadModeElementType>   cuFpBndIdxTex;


__device__
size_t findLowerBound(hashtype val, u_int32_t lb, u_int32_t ub)
{
    int32_t distance = ub;
    distance -= lb;
    while (distance > 0)
    {
        u_int32_t current = lb;
        u_int32_t step = distance / 2;
        current += step;
        hashtype pt = tex1Dfetch(cuFpCoordsTex, current);
        if (pt < val) {
            lb = current+1;
            distance -= step + 1;
        } else {
            distance = step;
        }

    }
    hashtype found = tex1Dfetch(cuFpCoordsTex, lb);
    if (found != val)
        return CU_INVALID_INDEX_ITEM;
    return lb;
}

__device__
size_t findUpperBound(hashtype val, u_int32_t lb, u_int32_t ub)
{
    int32_t distance = ub;
    distance -= lb;
    while (distance > 0)
    {
        u_int32_t current = lb;
        u_int32_t step = distance / 2;
        current += step;
        hashtype pt = tex1Dfetch(cuFpCoordsTex, current);
        if (pt <= val) {
            lb = current+1;
            distance -= step + 1;
        } else {
            distance = step;
        }
    }
    return lb;
}

__host__ __device__
static inline hashtype get_hash_mask(size_t spinquadCount, int sq_j, size_t parts, int part_i)
{
    //std::cerr << __PRETTY_FUNCTION__ << " " << spinquadCount << " " << sq_j << " " << parts << " " << part_i << " ";

    hashtype retval = 0;
    size_t offset = sq_j % HASHBITS;
    size_t currentPart = sq_j / HASHBITS;

    if (part_i == currentPart)
        retval = (1 << offset);

    //std::cerr << "RV: " << retval << std::endl;
    return retval;
}


static const size_t SPINQUADSETP = 4;
#define APPLY_VECTORIZED_STUFF \
    X(1) \
    X(2) \
    X(3) \
    X(4)

/* \ \
    X(5) \
    X(6) \
    X(7) \
    X(8)*/

struct FindItem
{
    size_t spinquadCount;
    int sq_j;
    size_t parts;
    int part_i;
    size_t toCount;

    FindItem(size_t _spinquadCount, int _sq_j, size_t _parts, int _part_i, size_t toBeCounted)
        : spinquadCount(_spinquadCount),
          sq_j(_sq_j),
          parts(_parts),
          part_i(_part_i),
          toCount(toBeCounted)
    {
    }

    __device__
    void compute(hashtype hash,
                 u_int32_t lb, u_int32_t& lb_out,
                 u_int32_t ub, u_int32_t& ub_out)
    {
        if (lb == CU_INVALID_INDEX_ITEM || ub == CU_INVALID_INDEX_ITEM)
            return;

        lb = findLowerBound(hash, lb, ub);
        if (lb == CU_INVALID_INDEX_ITEM)
            ub = CU_INVALID_INDEX_ITEM;
        else //find upper bound - it must exists
            ub = findUpperBound(hash, lb, ub);

        lb_out = lb;
        ub_out = ub;
    }

    template<typename Tuple>
    __device__
    void operator()(Tuple t)
    {
        const u_int32_t index = thrust::get<0>(t); // index
        const hashtype hash = tex1Dfetch(cuFpCoordsTex, index);
        hashtype mask;

#define X(iter) \
        if (toCount >= iter) { \
        mask = get_hash_mask(spinquadCount, sq_j+iter, parts, part_i); \
        compute(mask ^ hash, \
                thrust::get<2*iter-1>(t), thrust::get<2*iter-1>(t), \
                thrust::get<2*iter-0>(t), thrust::get<2*iter-0>(t)); \
        }
        APPLY_VECTORIZED_STUFF
#undef X
    }
};

struct PairIndexCleaner
{
    __device__
    bool operator()(u_int32_t index)
    {
        u_int32_t val = tex1Dfetch(cuFpBndIdxTex, index);
        return CU_INVALID_INDEX_ITEM == val;
    }
};

struct PairIndexCopier
{
    __device__
    u_int32_t operator()(u_int32_t index)
    {
        return tex1Dfetch(cuFpBndIdxTex, index);
    }
};

static void extract_result(size_t spins_size,
                           const thrust::device_vector<u_int32_t>& lower_bound,
                           thrust::device_vector<u_int32_t>& output1,
                           thrust::device_vector<u_int32_t>& output2)
{
    thrust::device_vector<u_int32_t> indices(spins_size);
    thrust::sequence(indices.begin(), indices.end());

    thrust::device_vector<u_int32_t> counterPairs;

    cutilSafeCall( cudaBindTexture(NULL, cuFpBndIdxTex,
                                   thrust::raw_pointer_cast(lower_bound.data()),
                                   sizeof(u_int32_t)*lower_bound.size()) );

    indices.resize(thrust::remove_if(indices.begin(), indices.end(), PairIndexCleaner()) - indices.begin());
    counterPairs.resize(indices.size());
    thrust::transform(indices.begin(), indices.end(), counterPairs.begin(), PairIndexCopier());
    cutilSafeCall( cudaUnbindTexture(cuFpBndIdxTex) );

    size_t currentSize = output1.size();
    output1.resize(output1.size() + indices.size());
    output2.resize(output2.size() + counterPairs.size());
    thrust::copy(indices.begin(), indices.end(), output1.begin()+currentSize);
    thrust::copy(counterPairs.begin(), counterPairs.end(), output2.begin()+currentSize);
}

void locate_pairs(thrust::device_vector<float4>& spins,
                  const std::vector<float>& spinquadrics,
                  size_t spinquadCount,
                  std::vector<u_int32_t>& output1,
                  std::vector<u_int32_t>& output2)
{
    //std::cerr << "SPINQUADS: " << spinquadCount << "\n";

    thrust::device_vector<u_int32_t> d_output1;
    thrust::device_vector<u_int32_t> d_output2;

#define X(iter) \
    thrust::device_vector<u_int32_t> lower_bound_##iter(spins.size()); \
    thrust::device_vector<u_int32_t> upper_bound_##iter(spins.size());
    APPLY_VECTORIZED_STUFF
#undef X

    size_t parts = inc_div<size_t>(spinquadrics.size()/10, HASHBITS);
    std::vector<size_t> partsSizes(parts);

    size_t rem = spinquadCount;
    for (int i=0;i<parts; ++i)
    {
        partsSizes[i] = std::min<size_t>(rem, HASHBITS);
        rem -= HASHBITS;
    }
#ifdef USE_HUGE_GPU_MEM
    std::vector<thrust::device_vector<hashtype> > hashPart(parts, thrust::device_vector<hashtype>(spins.size()));

    for (int i=0;i<parts; ++i)
    {
        compute_hash_part(spins, spinquadrics, hashPart[i], i, partsSizes[i]);
    }
#else
    thrust::device_vector<hashtype> hashPart(spins.size());
#endif

    //for each spinquad:
    for (int j=0;j<spinquadCount;j+=SPINQUADSETP)
    {
        size_t spinQuadsToCompute = std::min<size_t>(spinquadCount - j, SPINQUADSETP);
        //std::cerr << "spinQuadsToCompute: " << spinQuadsToCompute << std::endl;

#define X(iter) \
    thrust::fill(lower_bound_##iter.begin(), lower_bound_##iter.end(), 0); \
    thrust::fill(upper_bound_##iter.begin(), upper_bound_##iter.end(), spins.size());
    APPLY_VECTORIZED_STUFF
#undef X
        for (int i=parts-1;i>=0; --i)
        {
#ifndef USE_HUGE_GPU_MEM
            compute_hash_part(spins, spinquadrics, hashPart, i, partsSizes[i]);
            cutilSafeCall( cudaBindTexture(NULL, cuFpCoordsTex, thrust::raw_pointer_cast(hashPart.data()), sizeof(hashtype)*hashPart.size()) );
#else
            cutilSafeCall( cudaBindTexture(NULL, cuFpCoordsTex, thrust::raw_pointer_cast(hashPart[i].data()), sizeof(hashtype)*hashPart[i].size()) );
#endif

            thrust::for_each(
                        thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<u_int32_t>(0),
                                                         #define X(iter) \
                                                                     lower_bound_##iter.begin(), \
                                                                     upper_bound_##iter.begin(),
                                                             APPLY_VECTORIZED_STUFF
                                                         #undef X
                                                                     thrust::counting_iterator<u_int32_t>(0)
                                                                     )),
                        thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<u_int32_t>(spins.size()),
                                                         #define X(iter) \
                                                                     lower_bound_##iter.end(), \
                                                                     upper_bound_##iter.end(),
                                                             APPLY_VECTORIZED_STUFF
                                                         #undef X
                                                                     thrust::counting_iterator<u_int32_t>(spins.size())
                                                                     )),
                        FindItem(spinquadCount, j, parts, i, spinQuadsToCompute));

            cutilSafeCall( cudaUnbindTexture(cuFpCoordsTex) );
        }
        //Collect results
        {

#define X(iter) \
    if (spinQuadsToCompute >= iter) extract_result(spins.size(), lower_bound_##iter, d_output1, d_output2);
    APPLY_VECTORIZED_STUFF
#undef X
        }
    }
    output1.resize(d_output1.size());
    thrust::copy(d_output1.begin(), d_output1.end(), output1.begin());
    output2.resize(d_output2.size());
    thrust::copy(d_output2.begin(), d_output2.end(), output2.begin());
    //std::cerr << "COUNT: " << finalCount << std::endl;
    //std::cerr << "SIZES: " << output1.size() << " " << output2.size() << std::endl;


}

}
