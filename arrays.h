#pragma once
#include <vector>
#include <cassert>
#include <string.h>

typedef int ae_int_t;

class real_2d_array
{
private:
    std::vector<std::vector<float>> m_data;

public:
    void setlength(int rows, int cols)
    {
        m_data.resize(rows);
        for (auto& col : m_data)
            col.resize(cols);
    }

    int rows() const
    {
        return m_data.size();
    }

    int cols() const
    {
        if (m_data.size() == 0)
            return 0;
        assert(m_data.size() > 0);
        return m_data[0].size();
    }

    float& operator()(int row, int col)
    {
        return m_data[row][col];
    }

    float* operator[](int row)
    {
        return m_data[row].data();
    }

    const float& operator()(int row, int col) const
    {
        return m_data[row][col];
    }

    const float* operator[](int row) const
    {
        return m_data[row].data();
    }
};

class integer_2d_array
{
private:
    std::vector<std::vector<int>> m_data;

public:
    void setlength(int rows, int cols)
    {
        m_data.resize(rows);
        for (auto& col : m_data)
            col.resize(cols);
    }

    int rows() const
    {
        return m_data.size();
    }

    int cols() const
    {
        if (m_data.size() == 0)
            return 0;
        assert(m_data.size() > 0);
        return m_data[0].size();
    }

    int& operator()(int row, int col)
    {
        return m_data[row][col];
    }

    int* operator[](int row)
    {
        return m_data[row].data();
    }

    const int& operator()(int row, int col) const
    {
        return m_data[row][col];
    }

    const int* operator[](int row) const
    {
        return m_data[row].data();
    }
};

class real_1d_array
{
private:
    std::vector<float> m_data;

public:
    void setcontent(int size, const float* data)
    {
        m_data.resize(size);
        // copy
        memcpy(m_data.data(), data, size * sizeof(float));
    }

    float* getcontent()
    {
        return m_data.data();
    }

    const float* getcontent() const
    {
        return m_data.data();
    }

    int length() const
    {
        return m_data.size();
    }

    void setlength(int size)
    {
        m_data.resize(size);
    }

    float& operator()(int i)
    {
        return m_data[i];
    }

    float& operator[](int i)
    {
        return m_data[i];
    }

    const float& operator()(int i) const
    {
        return m_data[i];
    }

    const float& operator[](int i) const
    {
        return m_data[i];
    }
};

class integer_1d_array
{
private:
    std::vector<int> m_data;

public:
    void setcontent(int size, const int* data)
    {
        m_data.resize(size);
        // copy
        memcpy(m_data.data(), data, size * sizeof(int));
    }

    int* getcontent()
    {
        return m_data.data();
    }

    const int* getcontent() const
    {
        return m_data.data();
    }

    void setlength(int size, int defaultValue = 0)
    {
        m_data.resize(size, defaultValue);
    }

    int length() const
    {
        return m_data.size();
    }

    int& operator()(int i)
    {
        return m_data[i];
    }

    int& operator[](int i)
    {
        return m_data[i];
    }

    const int& operator()(int i) const
    {
        return m_data[i];
    }

    const int& operator[](int i) const
    {
        return m_data[i];
    }
};

void write_data(const char* file_name, const real_2d_array& data);

void write_data(const char* file_name, const integer_1d_array& data);

void print_array(const real_1d_array& arr);

void print_array(const integer_1d_array& arr);

void print_array(const real_2d_array& arr);

void print_array(const integer_2d_array& arr);

std::vector<real_1d_array> to_1d_arrays(const real_2d_array& data);

void print_c_array(int* arr, int size);

void print_vector(std::vector<int> v);

void print_c_array(float* arr, int size);

void print_vector(std::vector<float> v);

void knncuda_idx_to_2d_array(int* arr, int num_points, int num_nn, integer_2d_array& arr2d);

void knncuda_dist_to_2d_array(float* arr, int num_points, int num_nn, real_2d_array& arr2d);

void knncuda_idx_to_2d_array(int* arr, int num_points, int num_nn, integer_2d_array& arr2d, int numPointsToSkipInArr);

void knncuda_dist_to_2d_array(float* arr, int num_points, int num_nn, real_2d_array& arr2d, int numPointsToSkipInArr);