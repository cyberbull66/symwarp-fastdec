#pragma once

#include <tuple>

std::tuple<int, int> get_rows_coloumns(const char* file_name);

void read_data(const char* file_name, float* data, int num_points, int num_dims);

void print_c_array_2d(int* arr, int rows, int columns, const char* notes = "");

void print_c_array_2d(float* arr, int rows, int columns, const char* notes = "");

void print_dist_2d(float* arr, int rows, int columns, const char* notes = "kNN Distances");

void print_index_2d(int* arr, int rows, int columns, const char* notes = "kNN Indexes");

void Transpose(float* arr, int nPoints, int nDims, float* transposedArr);

template<class ForwardIt>
ForwardIt min_element(ForwardIt first, ForwardIt last)
{
    if (first == last)
        return last;
 
    ForwardIt smallest = first;
    ++first;
 
    for (; first != last; ++first)
        if (*first < *smallest)
            smallest = first;
 
    return smallest;
}

template<class ForwardIt>
ForwardIt max_element(ForwardIt first, ForwardIt last)
{
    if (first == last)
        return last;
 
    ForwardIt largest = first;
    ++first;
 
    for (; first != last; ++first)
        if (*largest < *first)
            largest = first;
 
    return largest;
}

