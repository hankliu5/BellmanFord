/*
 * This is a CUDA version of bellman_ford algorithm
 * Compile: nvcc -std=c++11 -arch=sm_52 -o cuda_bellman_ford cuda_bellman_ford.cu
 * Run: ./cuda_bellman_ford <input file> <number of blocks per grid> <number of threads per block>, you will find the output file 'output.txt'
 * */

#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <sys/time.h>
#include <climits>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// for timing
#include <sys/time.h>

using std::string;
using std::cout;
using std::endl;

#define INF INT_MAX
#define THREADS_PER_BLOCK 1024

/*
 * This is a CHECK function to check CUDA calls
 */
#define CHECK(call)                                                            \
{                                                                              \
	const cudaError_t error = call;                                            \
	if (error != cudaSuccess)                                                  \
	{                                                                          \
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
		fprintf(stderr, "code: %d, reason: %s\n", error,                       \
				cudaGetErrorString(error));                                    \
				exit(1);                                                       \
	}                                                                          \
}

void get_outedges(int64_t* graph, int64_t* outedges, size_t interval_st, size_t num_of_subvertices, size_t num_of_vertices) {
    memcpy(outedges, (graph + interval_st * num_of_vertices), sizeof(int64_t) * num_of_subvertices * num_of_vertices);
}

void get_inedges(int64_t* graph, int64_t* inedges, size_t interval_st, size_t interval_en, size_t num_of_vertices) {
    size_t i, j, row;
    int64_t *graph_ptr, *inedges_ptr;
    // in column favor but transpose into row.
    for (i = 0; i < num_of_vertices; i++) {
        for (row = 0, j = interval_st; j < interval_en; row++, j++) {
            // printf("i: %lu, j: %lu, row: %lu\n", i, j, row);
            *(inedges + row * num_of_vertices + i) = *(graph + i * num_of_vertices + j);
        }
    }
}

__global__ void bellman_ford_one_iter(size_t n, int64_t *d_mat, int64_t *d_dist, bool *d_has_next){
	size_t global_tid = blockDim.x * blockIdx.x + threadIdx.x;
	size_t v = global_tid;
	size_t u;
	if (global_tid >= n) return;
	for(u = 0; u < n; u++){
		int64_t weight = d_mat[u * n + v]; // row is src, col is dst
		if (weight > 0) {
			int64_t new_dist = d_dist[u] + weight;
			if(new_dist < d_dist[v]){
				d_dist[v] = new_dist;
				*d_has_next = true;
			}
		}
	}
}

/**
 * Bellman-Ford algorithm. Find the shortest path from vertex 0 to other vertices.
 * @param blockPerGrid number of blocks per grid
 * @param threadsPerBlock number of threads per block
 * @param n input size
 * @param *mat input adjacency matrix
 * @param *dist distance array
 * @param *has_negative_cycle a bool variable to recode if there are negative cycles
 */
void bellman_ford(size_t n, int64_t *mat, int64_t *dist, bool *has_negative_cycle) {
	size_t iter_num = 0;
	int64_t *d_mat, *d_dist;
	bool *d_has_next, h_has_next;
	size_t i;

	cudaMalloc(&d_mat, sizeof(int64_t) * n * n);
	cudaMalloc(&d_dist, sizeof(int64_t) * n);
	cudaMalloc(&d_has_next, sizeof(bool));

	*has_negative_cycle = false;

	for(i = 0 ; i < n; i++){
		dist[i] = INF;
	}

	dist[0] = 0;
	cudaMemcpy(d_mat, mat, sizeof(int64_t) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dist, dist, sizeof(int64_t) * n, cudaMemcpyHostToDevice);

	do {
		h_has_next = false;
		cudaMemcpy(d_has_next, &h_has_next, sizeof(bool), cudaMemcpyHostToDevice);

		bellman_ford_one_iter<<<(n+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(n, d_mat, d_dist, d_has_next);
		CHECK(cudaDeviceSynchronize());
		cudaMemcpy(&h_has_next, d_has_next, sizeof(bool), cudaMemcpyDeviceToHost);

		iter_num++;
		if(iter_num >= n-1){
			*has_negative_cycle = true;
			break;
		}
	} while (h_has_next);

	if(! *has_negative_cycle){
		cudaMemcpy(dist, d_dist, sizeof(int64_t) * n, cudaMemcpyDeviceToHost);
	}

	cudaFree(d_mat);
	cudaFree(d_dist);
	cudaFree(d_has_next);
}

/**
 * TODO section:
 * maybe we can borrow the log system from graphchi?
 */
int main(int argc, char** argv) {
    int64_t *graph, *outedges, *inedges;
    int fd;
    size_t num_of_vertices, num_of_subvertices, niters;
    size_t iter, st, i;
    
    // result
    int64_t *vertices;
	bool has_negative_cycle = false;

    // timing
    struct timeval h_start, h_end;
    long duration;

    if (argc < 4) {
        printf("usage: %s <graph path> <# of vertices> <# of subvertices>\n", argv[0]);
        exit(1);
    } 

    // I/O part, open in mmap mode
    fd = open(argv[1], O_RDONLY);
    num_of_vertices = (size_t) atoi(argv[2]);
    num_of_subvertices = (size_t) atoi(argv[3]);
    niters = (size_t) atoi(argv[4]);
    graph = (int64_t *) mmap(NULL, sizeof(int64_t) * num_of_vertices * num_of_vertices, PROT_READ, MAP_PRIVATE, fd, 0);

    // calculate the largest stripe we can have
    // Assume we have 1 GB (like graphchi), and at least we can contain one row and one column of the graph
    // and assume we those numbers are the power of 2
    const size_t stripe_sz = num_of_vertices * num_of_subvertices * sizeof(int64_t);
    // num_of_subvertices = 32;
    printf("num_of_subvertices: %lu\n", num_of_subvertices);

    // subgraph initialization
    outedges = (int64_t *) malloc(stripe_sz);
    inedges = (int64_t *) malloc(stripe_sz);
    printf("graph: %p, outedges: %p, inedges: %p\n", graph, outedges, inedges);

    // PR initialization
    vertices = (int64_t *) calloc(sizeof(int64_t), num_of_vertices);
    
    for (i = 0; i < num_of_vertices; i++) {
        vertices[i] = INF;
    }

	bellman_ford(num_of_vertices, graph, vertices, &has_negative_cycle);

    FILE *fp = fopen("log.txt", "w");
    for (i = 0; i < num_of_vertices; i++) {
        fprintf(fp, "%lu %lu\n", i, vertices[i]);
    }
    fclose(fp);
    // cleanup
    munmap(graph, sizeof(int64_t) * num_of_vertices * num_of_vertices);
    close(fd);

    free(outedges);
    free(inedges);
    free(vertices);

    return 0;
}