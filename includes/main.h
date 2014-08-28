#pragma once

#include <vector>
#include <vector_types.h>
#include <sys/types.h>

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

void cucs_entry();

