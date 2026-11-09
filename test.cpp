#include <algorithm>
#include <climits>
#include <math.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "helper.h"
#include "symwarpfastdec.h"
#if USE_KNNCUDA
#include "knncudafastdec.h"
#endif
#if USE_FAISS
#include "faissgpu-fastdec/faissgpufastdec.h"
#endif
#include "full_gpu/symwarpfullgpufastdec.h"
#include "elapsedtime.h"
#include "textreader.h"

#if defined(_WIN32)
#include <chrono>

typedef struct timeval {
    long long tv_sec;
    long long tv_usec;
} timeval;

struct timezone {
    int tz_minuteswest;
    int tz_dsttime;
};

int gettimeofday(struct timeval* tp, struct timezone* /*tzp*/) {
    namespace sc = std::chrono;
    sc::system_clock::duration d = sc::system_clock::now().time_since_epoch();
    sc::seconds s = sc::duration_cast<sc::seconds>(d);
    tp->tv_sec = s.count();
    tp->tv_usec = sc::duration_cast<sc::microseconds>(d - s).count();
    return 0;
}

typedef std::chrono::high_resolution_clock::time_point time_var;
#define duration(a) std::chrono::duration_cast<std::chrono::nanoseconds>(a).count()
#define time_now() std::chrono::high_resolution_clock::now()

#else

#include <sys/time.h>

#endif // _WIN32

enum class DecBackend
{
    SymWarp,
    KnnCuda,
    FaissGpu,
    FullGpu
};

int RunFastDEC(const char* processed_data_file,
               int num_k_points,
               int num_clusters,
               int num_runs,
               DecBackend backend,
               int cuda_device_id)
{
    SystemTimer timer;
    timer.Start();

    auto num_rows_columns = get_rows_coloumns(processed_data_file);

    // Parameters
    const int ref_nb = std::get<0>(num_rows_columns); 
    const int query_nb = ref_nb; 
    const int dim = std::get<1>(num_rows_columns); 
    const int k = num_k_points; 

    // Display
    printf("PARAMETERS\n");
    printf("- Dataset file: %s\n", processed_data_file);
    printf("- Number reference points : %d\n", ref_nb);
    printf("- Number query points     : %d\n", query_nb);
    printf("- Dimension of points     : %d\n", dim);
    printf("- Number of neighbors     : %d\n\n", k);

    // Sanity check
    if (ref_nb < k) {
        printf("Error: k value is larger that the number of reference points\n");
        return EXIT_FAILURE;
    }

    // Allocate input points and output k-NN distances / indexes
    float* ref = (float*)malloc((size_t)ref_nb * dim * sizeof(float));
    float* query = (float*)malloc((size_t)query_nb * dim * sizeof(float));
    float* knn_dist = (float*)malloc((size_t)query_nb * k * sizeof(float));
    int* knn_index = (int*)malloc((size_t)query_nb * k * sizeof(int));

    // Allocation checks
    if (!ref || !query || !knn_dist || !knn_index) {
        printf("Error: Memory allocation error\n");
        free(ref);
        free(query);
        free(knn_dist);
        free(knn_index);
        return EXIT_FAILURE;
    }

    read_data(processed_data_file, ref, ref_nb, dim);

    float* ref_d = (float*)malloc((size_t)ref_nb * dim * sizeof(float));
    Transpose(ref, ref_nb, dim, ref_d);
    memcpy(ref, ref_d, (size_t)ref_nb * dim * sizeof(float));
    free(ref_d);

    memcpy(query, ref, (size_t)ref_nb * dim * sizeof(float));

    timer.Stop("Finished preparation of data including Transpose");
    timer.Start(false);

    for (int run = 1; run <= num_runs; ++run)
    {
        auto run_fit_and_report = [&](auto& fastDec, const char* name)
        {
            SystemTimer fastdec_timer;
            fastdec_timer.Start();
            fastDec.Fit(ref, ref_nb, dim);

            char marker[64];
            snprintf(marker, sizeof(marker), "Finished run %d of %s-GPU", run, name);
            fastdec_timer.Stop(marker);

            fastdec_timer.Print();
            printf("\n");

            printf("Cluster centers found: \n");
            print_array(fastDec.get_center_indices());
            printf("\n");
        };

        if (backend == DecBackend::SymWarp)
        {
            SymWarpFastDEC fastDec(num_k_points, num_clusters, "symtri_knn_cuda", cuda_device_id);
            run_fit_and_report(fastDec, "SymWarpFastDEC");
        }
#if USE_KNNCUDA
        else if (backend == DecBackend::KnnCuda)
        {
            KnnCudaFastDEC fastDec(num_k_points, num_clusters, "knn_cuda_global", cuda_device_id);
            run_fit_and_report(fastDec, "KnnCudaFastDEC");
        }
#endif
#if USE_FAISS
        else if (backend == DecBackend::FaissGpu)
        {
            FaissGpuFastDEC fastDec(num_k_points, num_clusters, "faiss_self_knn_cuda", cuda_device_id);
            run_fit_and_report(fastDec, "FaissGpuFastDEC");
        }
#endif
        else if (backend == DecBackend::FullGpu)
        {
            SymWarpFullGpuFastDEC fastDec(num_k_points, num_clusters, "symtri_knn_cuda_full_gpu", cuda_device_id);
            run_fit_and_report(fastDec, "FullGpuFastDEC");
        }
        else
        {
            printf("Error: unsupported backend enum value %d in RunFastDEC\n", (int)backend);
            return EXIT_FAILURE;
        }
    }

    timer.Stop("Finished all runs");
    timer.Print();

    // Deallocate memory 
    free(ref);
    free(query);
    free(knn_dist);
    free(knn_index);

    return EXIT_SUCCESS;
}

static void print_usage(const char* prog)
{
	printf("Usage: %s [--gpu N] [--backend symwarp|knncuda|faissgpu|fullgpu] [config_file]\n", prog);
	printf("  --gpu N, -g N  index of the CUDA device to use (default: 0)\n");
	printf("  --backend, -b  clustering backend: symwarp (default), knncuda, faissgpu, or fullgpu\n");
	printf("                 (knncuda requires a build with USE_KNNCUDA=1, faissgpu with USE_FAISS=1)\n");
	printf("  config_file    path to the config file (default: ./config.txt)\n");
}

static bool parse_gpu_id(const char* text, int* gpu_id)
{
	char* end = nullptr;
	long value = strtol(text, &end, 10);
	if (end == text || *end != '\0' || value < 0 || value > INT_MAX)
		return false;
	*gpu_id = (int)value;
	return true;
}

int main(int argc, char *argv[]) 
{
    std::string configFilePath = "./config.txt";
    DecBackend backend = DecBackend::SymWarp;
    int cuda_device_id = 0;
	for (int i = 1; i < argc; ++i)
	{
		const char* arg = argv[i];
		const char* gpu_text = nullptr;
        const char* backend_text = nullptr;
		if (strcmp(arg, "--gpu") == 0 || strcmp(arg, "-g") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Error: %s requires a device index\n", arg);
				print_usage(argv[0]);
				return EXIT_FAILURE;
			}
			gpu_text = argv[++i];
		}
		else if (strncmp(arg, "--gpu=", 6) == 0)
		{
			gpu_text = arg + 6;
		}
		else if (strcmp(arg, "--backend") == 0 || strcmp(arg, "-b") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Error: %s requires one of: symwarp, knncuda, faissgpu, fullgpu\n", arg);
				print_usage(argv[0]);
				return EXIT_FAILURE;
			}
			backend_text = argv[++i];
		}
		else if (strncmp(arg, "--backend=", 10) == 0)
		{
			backend_text = arg + 10;
		}
		else if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0)
		{
			print_usage(argv[0]);
			return EXIT_SUCCESS;
		}
		else
		{
			configFilePath = arg;
			continue;
		}

        if (gpu_text != nullptr)
        {
            if (!parse_gpu_id(gpu_text, &cuda_device_id))
            {
                printf("Error: invalid GPU index '%s'\n", gpu_text);
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
        }

        if (backend_text != nullptr)
        {
            if (strcmp(backend_text, "symwarp") == 0)
                backend = DecBackend::SymWarp;
            else if (strcmp(backend_text, "knncuda") == 0)
            {
#if USE_KNNCUDA
                backend = DecBackend::KnnCuda;
#else
                printf("Error: the knncuda backend is not built in. Rebuild with `make clean && make USE_KNNCUDA=1`.\n");
                return EXIT_FAILURE;
#endif
            }
            else if (strcmp(backend_text, "faissgpu") == 0)
            {
#if USE_FAISS
                backend = DecBackend::FaissGpu;
#else
                printf("Error: the faissgpu backend is not built in. Rebuild with `make clean && make USE_FAISS=1`.\n");
                return EXIT_FAILURE;
#endif
            }
            else if (strcmp(backend_text, "fullgpu") == 0)
                backend = DecBackend::FullGpu;
            else
            {
                printf("Error: invalid backend '%s'. Expected 'symwarp', 'knncuda', 'faissgpu', or 'fullgpu'.\n", backend_text);
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
        }
	}

	int nb_devices = 0;
	if (cudaGetDeviceCount(&nb_devices) != cudaSuccess || nb_devices == 0)
	{
		printf("Error: No CUDA device found\n");
		return EXIT_FAILURE;
	}
	if (cuda_device_id >= nb_devices)
	{
		printf("Error: GPU %d requested but only %d CUDA device(s) found (valid: 0-%d)\n",
			cuda_device_id, nb_devices, nb_devices - 1);
		return EXIT_FAILURE;
	}
	cudaDeviceProp prop;
	if (cudaGetDeviceProperties(&prop, cuda_device_id) == cudaSuccess)
		printf("Using GPU %d of %d: %s\n", cuda_device_id, nb_devices, prop.name);
	else
		printf("Using GPU %d of %d\n", cuda_device_id, nb_devices);

	auto lines = TextReader::Read(configFilePath.c_str());
	assert(lines.size() >= 3);
    auto data_file = lines[0];
	std::cout << "Read dataset from: " << data_file << std::endl;
	int num_k_points = stoi(lines[1]);
	std::cout << "Number of nearest neighbors: " << num_k_points << std::endl;
	int num_clusters = stoi(lines[2]);
	std::cout << "Number of clusters: " << num_clusters << std::endl;

	int num_runs = 1;
	if (lines.size() >= 4)
	{
		try
		{
			num_runs = stoi(lines[3]);
			if (num_runs < 1)
				num_runs = 1;
		}
		catch (const std::exception&)
		{
			num_runs = 1;
		}
	}
	std::cout << "Number of runs: " << num_runs << std::endl;
    const char* backend_name = "symwarp";
    if (backend == DecBackend::KnnCuda)
        backend_name = "knncuda";
    else if (backend == DecBackend::FaissGpu)
        backend_name = "faissgpu";
    else if (backend == DecBackend::FullGpu)
        backend_name = "fullgpu";
    std::cout << "Backend: " << backend_name << std::endl;

    try
    {
        return RunFastDEC(data_file.c_str(), num_k_points, num_clusters, num_runs, backend, cuda_device_id);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    return EXIT_FAILURE;
}
