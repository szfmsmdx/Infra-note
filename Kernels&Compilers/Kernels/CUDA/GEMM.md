# v1
```cpp
__global__ void gemm_v1(
	const float* a,
	const float* b,
	float*c,
	int m, int n, int k
){
	// a[m, k], b[k, n], c[m, n]
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(row < m && col < n){
		float sum = 0.0f;
		for (int i = 0; i < k; ++i) {
			sum += a[row * k + i] * b[i * n + col];
		}
		c[row * n + col] = sum;
	}
}
```

# v2
简述一下 xy 的优化
1. ABC行存储 
2. 矩阵乘拆成向量点乘 
3. warp 应该是先x变，然后y再变，他是以 x 为单位变，而且 warp 最多32，所以我们 x 设置为 32 为一个 warp 
4. 对于 C\[m,n]=A\[m,i] * B\[i,n] for i in k 
5. 对于这个 for，我们也不能认为他是先 for 的，而是多线程分发下去的 
6. 多线程分发的时候对于 B 来说是 \[i, n+0] B\[i,n+1]... 
7. 对 A 来说则是 \[Am+0,i] ... 
8. 此时如果把 x 分配给A，那么A就需要取下一行，每个线程之前的内存不连续 
9. 如果把 x 分配给B，是连续的，没问题 
10. 在 9 的情况下，再看 A，A拿到的是y，在B多线程的时候A是不变的，他会被warp内广播，所以访存不变

还有个更简单的思路是：
- C的列存储更为高效

那么这里把 block size 设置为分块的 tile size，可以得到一版代码：
```cpp
__global__ void matrix_multiplication_kernel_v2(
const float* A,
const float* B,
float* C,
int M, int N, int K
) {
	__shared__ float a[TILE_SIZE][TILE_SIZE];
	__shared__ float b[TILE_SIZE][TILE_SIZE];
	
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = blockIdx.y * TILE_SIZE + ty;
	int col = blockIdx.x * TILE_SIZE + tx;
	
	float sum = 0.0f;
	// 遍历 K 维度的 tile
	for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
	// A tile: [row, t * TILE_SIZE + tx]
		if (row < M && (t * TILE_SIZE + tx) < K) {
			a[ty][tx] = A[row * K + (t * TILE_SIZE + tx)];
		} else {
			a[ty][tx] = 0.0f;
	}
	
	// B tile: [t * TILE_SIZE + ty, col]
	if (col < N && (t * TILE_SIZE + ty) < K) {
		b[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
	} else {
		b[ty][tx] = 0.0f;
	}
	
	__syncthreads();
	
	#pragma unroll
	for (int k = 0; k < TILE_SIZE; ++k) {
		sum += a[ty][k] * b[k][tx];
	}
	
	__syncthreads();
	}
	
	if (row < M && col < N) {
		C[row * N + col] = sum;
	}
}
```

# v3

v2 版本使用了 share mem 来优化访存读写，但实际上取一次 share mem只做了一次计算，强度太低了