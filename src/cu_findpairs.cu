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

struct FindItem
{
    hashtype mask;

    FindItem(hashtype _) : mask(_)
    {
    }

    template<typename Tuple>
    __device__
    void operator()(Tuple t)
    {
        //All are u_int32_t, size_t
        u_int32_t index = thrust::get<0>(t); // index
        u_int32_t lb = thrust::get<1>(t); // lower bound
        u_int32_t ub = thrust::get<2>(t); // upper bound

        if (lb == CU_INVALID_INDEX_ITEM || ub == CU_INVALID_INDEX_ITEM)
            return;

        hashtype val = mask ^ tex1Dfetch(cuFpCoordsTex, index);

        lb = findLowerBound(val, lb, ub);
        if (lb == CU_INVALID_INDEX_ITEM)
            ub = CU_INVALID_INDEX_ITEM;
        else //find upper bound - it must exists
            ub = findUpperBound(val, lb, ub);

        thrust::get<1>(t) = lb;
        thrust::get<2>(t) = ub;
        thrust::get<3>(t) = ub - lb;
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

void locate_pairs(thrust::device_vector<float4>& spins,
                  std::vector<float>& spinquadrics, size_t spinquadCount)
{
    //std::cerr << "SPINQUADS: " << spinquadCount << "\n";

    std::vector<u_int32_t> output1;
    std::vector<u_int32_t> output2;

    thrust::device_vector<u_int32_t> lower_bound(spins.size());
    thrust::device_vector<u_int32_t> upper_bound(spins.size());
    thrust::device_vector<u_int32_t> differ_bound(spins.size());

    thrust::device_vector<hashtype> hashPart(spins.size());
    size_t parts = inc_div<size_t>(spinquadrics.size()/10, HASHBITS);

    thrust::fill(lower_bound.begin(), lower_bound.end(), 0);
    thrust::fill(upper_bound.begin(), upper_bound.end(), spins.size());

    std::vector<size_t> partsSizes(parts);

    size_t rem = spinquadCount;
    for (int i=0;i<parts; ++i)
    {
        partsSizes[i] = std::min<size_t>(rem, HASHBITS);
        rem -= HASHBITS;
    }
    //for each spinquad:
    size_t finalCount = 0;
    for (int j=0;j<spinquadCount;++j)
    {
        thrust::fill(lower_bound.begin(), lower_bound.end(), 0);
        thrust::fill(upper_bound.begin(), upper_bound.end(), spins.size());
        for (int i=parts-1;i>=0; --i)
        {
            get_hash_mask(spinquadCount, j, parts, i);
            compute_hash_part(spins, spinquadrics, hashPart, i, partsSizes[i]);

            cutilSafeCall( cudaBindTexture(NULL, cuFpCoordsTex,
                                           thrust::raw_pointer_cast(hashPart.data()), sizeof(hashtype)*hashPart.size()) );


            thrust::for_each(
                        thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<u_int32_t>(0),
                                                                     lower_bound.begin(),
                                                                     upper_bound.begin(),
                                                                     differ_bound.begin()
                                                                     )),
                        thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<u_int32_t>(spins.size()),
                                                                     lower_bound.end(),
                                                                     upper_bound.end(),
                                                                     differ_bound.begin()
                                                                     )),
                        FindItem(get_hash_mask(spinquadCount, j, parts, i)));

            cutilSafeCall( cudaUnbindTexture(cuFpCoordsTex) );
        }
        //TODO: Collect results
        //size_t count = lower_bound.size() - thrust::count(lower_bound.begin(), lower_bound.end(), CU_INVALID_INDEX_ITEM);
        //finalCount += count;

        {
            thrust::device_vector<u_int32_t> indices(spins.size());
            thrust::sequence(indices.begin(), indices.end());
            thrust::device_vector<u_int32_t> counterPairs;

            cutilSafeCall( cudaBindTexture(NULL, cuFpBndIdxTex,
                                           thrust::raw_pointer_cast(lower_bound.data()),
                                           sizeof(u_int32_t)*lower_bound.size()) );
            indices.resize(thrust::remove_if(indices.begin(), indices.end(), PairIndexCleaner()) - indices.begin());
            //std::cerr << "COUNT<X>: " << indices.size() << std::endl;
            counterPairs.resize(indices.size());
            thrust::transform(indices.begin(), indices.end(), counterPairs.begin(), PairIndexCopier());
            cutilSafeCall( cudaUnbindTexture(cuFpBndIdxTex) );

            size_t currentSize = output1.size();
            output1.resize(output1.size() + indices.size());
            output2.resize(output2.size() + counterPairs.size());
            thrust::copy(indices.begin(), indices.end(), output1.begin()+currentSize);
            thrust::copy(counterPairs.begin(), counterPairs.end(), output2.begin()+currentSize);
        }
    }
    std::cerr << "COUNT: " << finalCount << std::endl;
    std::cerr << "SIZES: " << output1.size() << " " << output2.size() << std::endl;
//    std::cerr << "Biggest diff: " <<
//                 *(thrust::max_element(differ_bound.begin(), differ_bound.end()))
//              << std::endl;

//    for (int i=0;i<10;++i)
//        //for (int i=0;i<spins.size();++i)
//    {
//        std::cerr << i << " " << std::bitset<32>(hashPart[i]) << " : "
//                  << lower_bound[i] << " - " << upper_bound[i]
//                     << " <> " << differ_bound[i]
//                        << " " << hashPart[i] << std::endl;
//    }
//    std::cerr << "=======================================" << std::endl;


}

}
