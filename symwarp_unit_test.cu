#include "symwarpfastdec.h"
#include "dcreductioncuda.h"
#include "dataset.h"
#include "helper.h"

#include <math.h>
#include <stdio.h>
#include <vector>

static bool approx_equal(float a, float b, float eps = 1e-5f)
{
    return fabsf(a - b) <= eps;
}

static int fail(const char* message)
{
    fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}

static int test_dc_reduction_known_case()
{
    const int ds_nb = 5;
    const int core_nb = 3;
    const int dim = 1;

    const float ds_host[ds_nb] = {0.0f, 2.0f, 5.0f, 10.0f, 20.0f};
    const float ds_den_host[ds_nb] = {1.0f, 3.0f, 2.0f, 5.0f, 4.0f};

    const int core_indices_host[core_nb] = {0, 2, 4};
    int cdh_index_host[core_nb] = {-9, -9, -9};
    int center_indices_host[1] = {-9};

    Dataset ds;
    ds.Bind(ds_host, ds_nb, dim);
    if (!warp_parallel_dc_reduction_cuda(ds, ds_den_host,
                                         core_indices_host, core_nb, 1,
                                         cdh_index_host, center_indices_host)) {
        return fail("warp_parallel_dc_reduction_cuda returned false for known-case input");
    }
    if (!ds.IsUploaded()) {
        return fail("warp_parallel_dc_reduction_cuda did not upload the dataset into the Dataset object");
    }
    ds.Release();

    const int expected_idx[core_nb] = {1, 1, 3};

    for (int i = 0; i < core_nb; ++i) {
        if (cdh_index_host[i] != expected_idx[i]) {
            return fail("warp_parallel_dc_reduction_cuda produced unexpected cdh_index");
        }
    }
    if (center_indices_host[0] != 4) {
        return fail("warp_parallel_dc_reduction_cuda produced unexpected center");
    }

    printf("PASS dc_reduction known-case outputs\n");
    return 0;
}

static int test_dc_reduction_no_cdh_case()
{
    const int ds_nb = 3;
    const int core_nb = 1;
    const int dim = 1;

    const float ds_host[ds_nb] = {1.0f, 2.0f, 3.0f};
    const float ds_den_host[ds_nb] = {10.0f, 9.0f, 8.0f};

    const int core_indices_host[core_nb] = {0};
    int cdh_index_host[core_nb] = {123};
    int center_indices_host[1] = {123};

    Dataset ds;
    ds.Bind(ds_host, ds_nb, dim);
    if (!warp_parallel_dc_reduction_cuda(ds, ds_den_host,
                                         core_indices_host, core_nb, 1,
                                         cdh_index_host, center_indices_host)) {
        return fail("warp_parallel_dc_reduction_cuda returned false for no-cdh input");
    }

    if (cdh_index_host[0] != -1 || center_indices_host[0] != 0) {
        return fail("warp_parallel_dc_reduction_cuda did not produce expected no-cdh outputs");
    }

    printf("PASS dc_reduction no-cdh sentinels\n");
    return 0;
}

static int test_fit_small_dataset()
{
    const int num_points = 8;
    const int num_dims = 2;

    const float row_major[num_points * num_dims] = {
        0.0f, 0.0f,
        0.1f, 0.2f,
        0.2f, -0.1f,
        -0.1f, 0.1f,
        10.0f, 10.0f,
        10.2f, 9.9f,
        9.8f, 10.1f,
        10.1f, 10.3f
    };

    std::vector<float> dim_major((size_t)num_points * num_dims);
    Transpose(const_cast<float*>(row_major), num_points, num_dims, dim_major.data());

    SymWarpFastDEC model(3, 2, "symwarp_unit_test_fit");
    model.Fit(dim_major.data(), num_points, num_dims);

    const integer_1d_array& centers = model.get_center_indices();
    const integer_1d_array& labels = model.get_labels();

    if (centers.length() != 2) {
        return fail("Fit produced unexpected number of centers");
    }
    if (labels.length() != num_points) {
        return fail("Fit produced unexpected number of labels");
    }

    bool has_left = false;
    bool has_right = false;
    for (int i = 0; i < centers.length(); ++i) {
        int center = centers[i];
        if (center < 0 || center >= num_points) {
            return fail("Fit produced out-of-range center index");
        }
        if (center < 4) has_left = true;
        else has_right = true;
    }
    if (!has_left || !has_right) {
        return fail("Fit centers do not cover both expected clusters");
    }

    for (int i = 0; i < labels.length(); ++i) {
        if (labels[i] < 0 || labels[i] >= 2) {
            return fail("Fit produced out-of-range label");
        }
    }

    printf("PASS SymWarpFastDEC Fit small dataset\n");
    return 0;
}

int main()
{
    if (test_dc_reduction_known_case() != 0) return 1;
    if (test_dc_reduction_no_cdh_case() != 0) return 1;
    if (test_fit_small_dataset() != 0) return 1;
    printf("PASS all unit tests\n");
    return 0;
}
