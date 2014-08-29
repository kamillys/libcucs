#pragma once

#include <vector>
#include <string>
#include <vector_types.h>
#include <sys/types.h>


/**
 *  Format of spinors:
 *   sum =
 *         (s0*v0+s1*v1+s2*v2+s3*v3) * v0 +
 *         (s1*v0+s4*v1+s5*v2+s6*v3) * v1 +
 *         (s2*v0+s5*v1+s7*v2+s8*v3) * v2 +
 *         (s3*v0+s6*v1+s8*v2+s9*v3) * v3;
 *  s[i] is matrix / spinquadric.
 *   This spinquadric is symetric and compressed in 10 floats instead of 16.
 *  v[i] is vector / spinor.
 *   Spinors are generated randomly.
 *
 */

std::string cucs_to_coord_string(const float4& spin,
                                 const std::vector<float>& data);

/**
 * @brief cucs_set_seed
 * @param seed
 */
void cucs_set_seed(unsigned long long seed);

/**
 * @brief cucs_compute_random_spinors
 * @param count
 * @return
 */
std::vector<float4> cucs_compute_random_spinors(size_t count);

/**
 * @brief cucs_compute_random_unique_spinors
 * @param spinquads
 * @param initialCout
 * @return
 */
std::vector<float4> cucs_compute_random_unique_spinors(
        const std::vector<float>& spinquads,
        size_t initialCout);

/**
 * @brief cucs_compute_unique_spinors
 * @param spinquads
 * @param spinors
 */
void cucs_compute_unique_spinors(
        const std::vector<float>& spinquads,
        std::vector<float4>& spinors);

/**
 * @brief cucs_compute_spinors_and_neighbours
 * @param spinquads
 * @param initialCout
 * @param spinors
 * @param neighbour_index_1
 * @param neighbour_index_2
 */
void cucs_compute_spinors_and_neighbours(
        const std::vector<float>& spinquads,
        size_t initialCout,
        /*out*/ std::vector<float4>& spinors,
        /*out*/ std::vector<u_int32_t>& neighbour_index_1,
        /*out*/ std::vector<u_int32_t>& neighbour_index_2);

/**
 * @brief cucs_compute_neighbours
 * @param spinquads
 * @param spinors
 * @param neighbour_index_1
 * @param neighbour_index_2
 */
void cucs_compute_neighbours(
        const std::vector<float>& spinquads,
        const std::vector<float4>& spinors,
        /*out*/ std::vector<u_int32_t>& neighbour_index_1,
        /*out*/ std::vector<u_int32_t>& neighbour_index_2);
