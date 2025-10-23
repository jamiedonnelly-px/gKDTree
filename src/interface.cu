#include <cuda_runtime.h>
#include <cukd/builder.h>
#include <cukd/knn.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <cmath>
#include <random>

namespace py = pybind11;

#define FIXED_RADIUS std::numeric_limits<float>::infinity()

using data_t = float3;
using data_traits = cukd::default_data_traits<float3>;

// CUDA KNN Kernel
template<int K>
__global__ void KnnKernelFixed(
    const float3* d_queries, 
    int numQueries,
    const cukd::SpatialKDTree<float3,data_traits> tree,
    int* d_indices,
    float* d_distances
){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numQueries) return;

    // Use FixedCandidateList for small k, HeapCandidateList for larger k
    using CandidateList = typename std::conditional<
        (K <= 10), 
        cukd::FixedCandidateList<K>, 
        cukd::HeapCandidateList<K>
    >::type;

    CandidateList result(FIXED_RADIUS);
    cukd::stackBased::knn<decltype(result), float3, data_traits> (result, tree, d_queries[tid]);

    for (int i = 0; i < K; i++) {
        int baseIdx = tid * K + i;
        int pointID = result.get_pointID(i);
        float distance = result.get_dist2(i);
        d_indices[baseIdx] = pointID;
        d_distances[baseIdx] = distance; 
    }
}

// Host dispatch function
void launchKnnKernel(
    const float3* d_queries, 
    int numQueries,
    const cukd::SpatialKDTree<float3, data_traits>& tree,
    int* d_indices, 
    float* d_distances, 
    int k
) {
    
    int threadsPerBlock = 256;
    int numBlocks = (numQueries + threadsPerBlock - 1) / threadsPerBlock;
    
    // Dispatch to template based on k value
    switch(k) {
        case 1:  KnnKernelFixed<1> <<<numBlocks, threadsPerBlock>>>(d_queries, numQueries, tree, d_indices, d_distances); break;
        case 2:  KnnKernelFixed<2> <<<numBlocks, threadsPerBlock>>>(d_queries, numQueries, tree, d_indices, d_distances); break;
        case 4:  KnnKernelFixed<4> <<<numBlocks, threadsPerBlock>>>(d_queries, numQueries, tree, d_indices, d_distances); break;
        case 8:  KnnKernelFixed<8> <<<numBlocks, threadsPerBlock>>>(d_queries, numQueries, tree, d_indices, d_distances); break;
        case 16: KnnKernelFixed<16><<<numBlocks, threadsPerBlock>>>(d_queries, numQueries, tree, d_indices, d_distances); break;
        case 32: KnnKernelFixed<32><<<numBlocks, threadsPerBlock>>>(d_queries, numQueries, tree, d_indices, d_distances); break;
        case 64: KnnKernelFixed<64><<<numBlocks, threadsPerBlock>>>(d_queries, numQueries, tree, d_indices, d_distances); break;
        case 128: KnnKernelFixed<128><<<numBlocks, threadsPerBlock>>>(d_queries, numQueries, tree, d_indices, d_distances); break;
        case 256: KnnKernelFixed<256><<<numBlocks, threadsPerBlock>>>(d_queries, numQueries, tree, d_indices, d_distances); break;
        default:
            throw std::runtime_error("Unsupported k value: " + std::to_string(k) + 
                                    ". Supported values: 1, 2, 4, 8, 16, 32, 64, 128, 256");
    }
    cudaDeviceSynchronize();
}

void knnSearchCuda(
    const float3* points, 
    const int numPoints,
    const float3* queries, 
    const int numQueries,
    const int k,
    int* output_indices,
    float* output_distances
) {

    // Ensure we can actually find k neighbors
    if (k > numPoints) {
        throw std::invalid_argument(
            "Requested k=" + std::to_string(k) + 
            " but only " + std::to_string(numPoints) + " points available"
        );
    }

    // Allocate device memory for points and queries; copy from host -> device
    float3* d_points;
    cudaMallocManaged(&d_points, numPoints * sizeof(float3));
    cudaMemcpy(d_points, points, numPoints * sizeof(float3), cudaMemcpyHostToDevice);
    
    float3* d_queries;
    cudaMallocManaged(&d_queries, numQueries * sizeof(float3));
    cudaMemcpy(d_queries, queries, numQueries * sizeof(float3), cudaMemcpyHostToDevice);
    
    // Allocate device memory for results
    int* d_indices;
    float* d_distances;
    cudaMallocManaged(&d_indices, numQueries * k * sizeof(int));
    cudaMallocManaged(&d_distances, numQueries * k * sizeof(float));

    // Build Spatial KD-Tree (managed memory)
    cukd::SpatialKDTree<float3, data_traits> tree;
    cukd::BuildConfig buildConfig{};
    buildTree(tree, d_points, numPoints, buildConfig);
    CUKD_CUDA_SYNC_CHECK();

    // Dispatch kernel and synchronise result
    launchKnnKernel(d_queries, numQueries, tree, d_indices, d_distances, k);
    cudaDeviceSynchronize();

    // Copy results device -> host with pre-allocate arrays.
    cudaMemcpy(output_indices, d_indices, numQueries * k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_distances, d_distances, numQueries * k * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_points);
    cudaFree(d_queries);
    cudaFree(d_indices);
    cudaFree(d_distances);
    cukd::free(tree);
}

std::tuple<py::array_t<int>, py::array_t<float>> 
knnSearchCudaNumPy(py::array_t<float> points_py, 
                   py::array_t<float> queries_py,
                   int k) {
    
    // Get input buffers (zero-copy access to NumPy data)
    auto points_buf = points_py.request();
    auto queries_buf = queries_py.request();
    
    int numPoints = points_buf.shape[0];
    int numQueries = queries_buf.shape[0];
    
    // Cast to your expected types (zero-copy)
    const float3* points = reinterpret_cast<const float3*>(points_buf.ptr);
    const float3* queries = reinterpret_cast<const float3*>(queries_buf.ptr);
    
    // Pre-allocate output NumPy arrays (avoid intermediate allocation)
    auto indices_py = py::array_t<int>({numQueries, k});    // Shape: (numQueries, k)
    auto distances_py = py::array_t<float>({numQueries, k}); // Shape: (numQueries, k)
    
    // Get direct pointers to NumPy memory
    int* indices_ptr = static_cast<int*>(indices_py.mutable_unchecked<2>().mutable_data(0, 0));
    float* distances_ptr = static_cast<float*>(distances_py.mutable_unchecked<2>().mutable_data(0, 0));
    
    // Your CUDA function - but modify it to write directly to NumPy arrays
    knnSearchCuda(points, numPoints, queries, numQueries, k, 
                       indices_ptr, distances_ptr);
    
    // Attempt to flush the device 
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return std::make_tuple(indices_py, distances_py);
}

PYBIND11_MODULE(_internal, mod) {
    mod.doc() = R"pbdoc(
        CUDA KD-Tree Python bindings
    )pbdoc";
    
    mod.def("knn_cuda", &knnSearchCudaNumPy,
        "Perform KNN search using CUDA",
        py::arg("points"), 
        py::arg("queries"), 
        py::arg("k"),
        py::return_value_policy::move  // For efficiency with large arrays
    );
}

