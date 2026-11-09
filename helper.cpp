#include "helper.h"
#include "arrays.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <cassert>

std::tuple<int, int> get_rows_coloumns(const char* file_name)
{
    int rows = 0;
    int commas = 0;
    std::ifstream file(file_name);
    std::string line;
    while (getline(file, line))
    {
        if (rows == 1)
        {
            
            for (auto c : line)
            {
                if (c == ',')
                    commas++;
            }
        }
        rows++;
    }
    return std::make_tuple(rows, commas + 1);
}

void read_data(const char* file_name, float* data, int num_points, int num_dims)
{
    const size_t max_line_length = (size_t)num_dims * 32 + 2;

    FILE* file = fopen(file_name, "r");
    char* content = (char*)malloc(max_line_length);
    if (!content) {
        fclose(file);
        return;
    }
    int row_count = 0;
    int col_count = 0;
    while (fgets(content, max_line_length, file))
    {
        char* v = strtok(content, ",");
        while (v)
        {
            char* ptr;
            float test = strtof(v, &ptr);
            
            data[(size_t)row_count * num_dims + col_count] = test;
            col_count++;

            v = strtok(NULL, ",");
        }
        row_count++;
        
        assert(row_count <= num_points);
        col_count = 0;
    }
    free(content);
    fclose(file);
}

void print_c_array_2d(int* arr, int rows, int columns, const char* notes)
{
    if (notes != "")
        printf("%s (%d, %d)\n", notes, rows, columns);
    printf("[");
    for (int i = 0; i < rows; ++i)
    {
        printf("(");
        for (int j = 0; j < columns; ++j)
        {
            if (j != 0)
                printf(",");
            printf("%d", arr[i * columns + j]);
        }
        printf(")\n");
    }
    printf("]\n");
}

void print_c_array_2d(float* arr, int rows, int columns, const char* notes)
{
    if (notes != "")
        printf("%s (%d, %d)\n", notes, rows, columns);
    printf("[");
    for (int i = 0; i < rows; ++i)
    {
        printf("(");
        for (int j = 0; j < columns; ++j)
        {
            if (j != 0)
                printf(",");
            printf("%f", arr[i * columns + j]);
        }
        printf(")\n");
    }
    printf("]\n");
}

void print_dist_2d(float* arr, int rows, int columns, const char* notes)
{
    if (notes != "")
        printf("%s (%d, %d)\n", notes, rows, columns);
    printf("[");
    for (int i = 0; i < rows; ++i)
    {
        printf("(");
        for (int j = 0; j < columns; ++j)
        {
            if (j != 0)
                printf(",");
            printf("%f", arr[j * rows + i]);
        }
        printf(")\n");
    }
    printf("]\n");
}

void print_index_2d(int* arr, int rows, int columns, const char* notes)
{
    if (notes != "")
        printf("%s (%d, %d)\n", notes, rows, columns);
    printf("[");
    for (int i = 0; i < rows; ++i)
    {
        printf("(");
        for (int j = 0; j < columns; ++j)
        {
            if (j != 0)
                printf(",");
            printf("%d", arr[j * rows + i]);
        }
        printf(")\n");
    }
    printf("]\n");
}

void Transpose(float* arr, int nPoints, int nDims, float* transposedArr)
{
    for (int i = 0; i < nPoints; ++i)
    {
        for (int d = 0; d < nDims; ++d)
        {
            *(transposedArr + (size_t)d * nPoints + i) = *(arr + (size_t)i * nDims + d);
        }
    }
}

void write_data(const char* file_name, const real_2d_array& data)
{
    std::ofstream myfile;
    myfile.open(file_name);
    for (int i = 0; i < data.rows(); ++i)
    {
        for (int j = 0; j < data.cols(); ++j)
        {
            myfile << data[i][j];
            if (j != data.cols() - 1)
                myfile << ",";
        }
        if (i != data.rows() - 1)
            myfile << "\n";
    }
    myfile.close();
}

void write_data(const char* file_name, const integer_1d_array& data)
{
    std::ofstream myfile;
    myfile.open(file_name);
    for (int i = 0; i < data.length(); ++i)
    {
        myfile << data[i];
        if (i != data.length() - 1)
            myfile << ",";
    }
    myfile.close();
}

void print_array(const real_1d_array& arr)
{
    printf("length: %d\n", arr.length());
    printf("[");
    for (int i = 0; i < arr.length(); i++)
    {
        if (i != 0)
            printf(",");
        printf("%f", arr[i]);
    }
    printf("]\n");
}

void print_array(const integer_1d_array& arr)
{
    printf("length: %d\n", arr.length());
    printf("[");
    for (int i = 0; i < arr.length(); i++)
    {
        if (i != 0)
            printf(",");
        printf("%d", arr[i]);
    }
    printf("]\n");
}

void print_array(const real_2d_array& arr)
{
    printf("rows: %d, cols: %d\n", arr.rows(), arr.cols());
    printf("[");
    for (int i = 0; i < arr.rows(); i++)
    {
        printf("[");
        for (int j = 0; j < arr.cols(); j++)
        {
            if (j != 0)
                printf(",");
            printf("%f", arr[i][j]);
        }
        printf("]");
        if (i != arr.rows() - 1)
            printf(",");
    }
    printf("]\n");
}

void print_array(const integer_2d_array& arr)
{
    printf("rows: %d, cols: %d\n", arr.rows(), arr.cols());
    printf("[");
    for (int i = 0; i < arr.rows(); i++)
    {
        printf("[");
        for (int j = 0; j < arr.cols(); j++)
        {
            if (j != 0)
                printf(",");
            printf("%d", arr[i][j]);
        }
        printf("]");
        if (i != arr.rows() - 1)
            printf(",");
    }
    printf("]\n");
}

std::vector<real_1d_array> to_1d_arrays(const real_2d_array& data)
{
    std::vector<real_1d_array> v_1d_array(data.rows());
    for (int i = 0; i < data.rows(); i++)
    {
        v_1d_array[i].setcontent(data.cols(), data[i]);
    }
    return v_1d_array;
}

void print_c_array(int* arr, int size)
{
    printf("length: %d\n", size);
    printf("[");
    for (int i = 0; i < size; ++i)
    {
        if (i != 0)
            printf(",");
        printf("%d", arr[i]);
    }
    printf("]\n");
}

void print_vector(std::vector<int> v)
{
    print_c_array(v.data(), v.size());
}

void print_c_array(float* arr, int size)
{
    printf("length: %d\n", size);
    printf("[");
    for (int i = 0; i < size; ++i)
    {
        if (i != 0)
            printf(",");
        printf("%f", arr[i]);
    }
    printf("]\n");
}

void print_vector(std::vector<float> v)
{
    print_c_array(v.data(), v.size());
}

void knncuda_idx_to_2d_array(int* arr, int num_points, int num_nn, integer_2d_array& arr2d)
{
    for (int i = 0; i < num_points; ++i)
    {
        auto row = arr2d[i];
        for (int j = 0; j < num_nn; ++j)
        {
            row[j] = arr[(size_t)j * num_points + i];
        }
    }
}

void knncuda_dist_to_2d_array(float* arr, int num_points, int num_nn, real_2d_array& arr2d)
{
    for (int i = 0; i < num_points; ++i)
    {
        auto row = arr2d[i];
        for (int j = 0; j < num_nn; ++j)
        {
            row[j] = arr[(size_t)j * num_points + i];
        }
    }
}

void knncuda_idx_to_2d_array(int* arr, int num_points, int num_nn, integer_2d_array& arr2d, int numPointsToSkipInArr)
{
    for (int i = 0; i < num_points; ++i)
    {
        auto row = arr2d[i];
        for (int j = 0; j < num_nn; ++j)
        {
            row[j] = arr[(size_t)(j + numPointsToSkipInArr) * num_points + i];
        }
    }
}

void knncuda_dist_to_2d_array(float* arr, int num_points, int num_nn, real_2d_array& arr2d, int numPointsToSkipInArr)
{
    for (int i = 0; i < num_points; ++i)
    {
        auto row = arr2d[i];
        for (int j = 0; j < num_nn; ++j)
        {
            row[j] = arr[(size_t)(j + numPointsToSkipInArr) * num_points + i];
        }
    }
}
