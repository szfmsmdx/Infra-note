# v1 naive GEMM
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

# v2 share mem GEMM
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

# v3 2D thread TILE GEMM

v2 版本使用了 share mem 来优化访存读写，但实际上取一次 share mem只做了一次计算，强度太低了，我们考虑每个线程负责一个部分，比如负责 TSIZE 个，那么此时线程数应该会减少，每个 block 就只有 TILE_SIZE / TSIZE 个线程了，grid 则不变

那么此时，我们还是要把元素给搬到 share mem 中，也就是说此时一个 share mem大小为 TILE_SIZE * TILE_SIZE，我们只有 TSIZE * TSIZE 个线程，所以每个线程负责 (TLS / TS) \*\* 2个元素

那么我们可以这么想：
- 对于一个 TLS x TLS 大小的块，我们直接遍历他的 id 即可，因为每次有 THREAD_PER_BLOCK x THREAD_PER_BLOCK 个线程，所以 id 的步长应该就是线程个数，每个步长我通过 tid 来控制就可以
- 那么我就可以找到这个 id 对应在 TLS x TLS 大小块中的行列，也就是：
	- int l_row = load_id / TILE_SIZE;
	- int l_col = load_id % TILE_SIZE;
- 那么对应我们就可以算出AB对应的行列

那么我们就完成了从 HBM 到 share mem的跨越，然后我们是做 share men和线程的对应，比如一个线程对应 TS x TS的小块，那么他应该是取 A 的一横条和 B 的一列条相乘，但是我们现在做的是外积，所以我们要取：
- A 横条中的一列一列
- B 列条中的一横一横

最后的代码如下：
```cpp
__global__ void matrix_multiplication_kernel_v3(
	const float* A,
	const float* B,
	float* C,
	int M, int N, int K
) {
	__shared__ float sa[TILE_SIZE][TILE_SIZE + 1]; // 避开 bank conflict
	__shared__ float sb[TILE_SIZE][TILE_SIZE];
	
	float ta[TSIZE];
	float tb[TSIZE];
	float tc[TSIZE][TSIZE] = {0.0f};
	
	int tx = threadIdx.x; // 0..3
	int ty = threadIdx.y; // 0..3
	
	// 线程对于 block 的起始位置偏移
	int row_start = ty * TSIZE; // 0,8,16,24
	int col_start = tx * TSIZE;
	
	// block 对于C的位置起始偏移
	int block_row = blockIdx.y * TILE_SIZE;
	int block_col = blockIdx.x * TILE_SIZE;
	
	int tid = ty * (TILE_SIZE / TSIZE) + tx; // 0..15
	
	for (int k_tile = 0; k_tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++k_tile) {
	
		#pragma unroll
		for (int i = 0; i < TILE_SIZE * TILE_SIZE; i += THREADS_PER_BLOCK * THREADS_PER_BLOCK){
			int load_id = i + tid;
			int l_row = load_id / TILE_SIZE;
			int l_col = load_id % TILE_SIZE;
			
			// 加载 A
			int g_row_a = block_row + l_row;
			int g_col_a = k_tile * TILE_SIZE + l_col;
			sa[l_row][l_col] = (g_row_a < M && g_col_a < K) ? A[g_row_a * K + g_col_a] : 0.0f;
			
			  
			// 加载 B
			int g_row_b = k_tile * TILE_SIZE + l_row;
			int g_col_b = block_col + l_col;
			sb[l_row][l_col] = (g_row_b < K && g_col_b < N) ? B[g_row_b * N + g_col_b] : 0.0f;
		}
	
	  
	__syncthreads();
	// shared -> register -> compute
	#pragma unroll
	for (int k = 0; k < TILE_SIZE; ++k) {
		// 加载 A 的寄存器：一个线程负责 TSIZE (8) 行
		#pragma unroll
		for (int i = 0; i < TSIZE; ++i)
			ta[i] = sa[ty * TSIZE + i][k];
		
		// 加载 B 的寄存器：一个线程负责 TSIZE (8) 列
		#pragma unroll
		for (int j = 0; j < TSIZE; ++j)
			tb[j] = sb[k][tx * TSIZE + j];
		
		#pragma unroll
		for (int i = 0; i < TSIZE; ++i) {
			#pragma unroll
			for (int j = 0; j < TSIZE; ++j) {
				tc[i][j] += ta[i] * tb[j];
			}
		}
	}
	
	__syncthreads();
	}
	
	#pragma unroll
	for (int i = 0; i < TSIZE; ++i) {
		int global_row = block_row + row_start + i;
		if (global_row < M) {
			#pragma unroll
			for (int j = 0; j < TSIZE; ++j) {
				int global_col = block_col + col_start + j;
				if (global_col < N) {
					C[global_row * N + global_col] = tc[i][j];
				}
			}
		}
	}
}
```