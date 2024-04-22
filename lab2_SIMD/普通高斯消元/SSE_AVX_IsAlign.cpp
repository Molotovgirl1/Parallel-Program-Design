#include<iostream>
#include<stdlib.h>
#include<Windows.h>
#include<nmmintrin.h> //SSSE4.2
#include<immintrin.h> //AVX、AVX
using namespace std;
const int N = 1024;
float m[N][N];
const int times = 10;
// 生成测试用例
void m_reset() {
	for (int i = 0;i < N;i++) {
		for (int j = 0;j < i;j++) {
			m[i][j] = 0;
		}
		m[i][i] = 1.0;
		for (int j = i + 1;j < N;j++) {
			m[i][j] = rand();
		}
	}
	for (int k = 0;k < N;k++) {
		for (int i = k + 1;i < N;i++) {
			for (int j = 0;j < N;j++) {
				m[i][j] += m[k][j];
			}
		}
	}
}
//普通高斯消元串行算法
void serial_LU() {
	for (int k = 0;k < N;k++) {
		for (int j = k + 1;j < N;j++) {
			m[k][j] /= m[k][k]; //将第k行对角线元素变为1
		}
		m[k][k] = 1.0;
		for (int i = k + 1;i < N;i++) {
			for (int j = k + 1;j < N;j++) {
				m[i][j] -= m[k][j] * m[i][k]; //从第i行消去第k行
			}
			m[i][k] = 0;
		}
	}
}
//对SSE两个部分进行并行化(不对齐)
void parallel_SSE_Unalign() {
	for (int k = 0;k < N;k++) {
		float tmp1[4] = { m[k][k] ,m[k][k] ,m[k][k] ,m[k][k] };
		__m128 tmp_kk = _mm_loadu_ps(tmp1);
		int num1 = k + 1;
		for (int j = k + 1;j + 4 <= N;j += 4, num1 = j) {
			__m128 tmp_kj = _mm_loadu_ps(m[k] + j);
			tmp_kj = _mm_div_ps(tmp_kj, tmp_kk);
			_mm_storeu_ps(m[k] + j, tmp_kj);
		}
		for (int j = num1;j < N;j++) {
			m[k][j] /= m[k][k];
		}
		m[k][k] = 1.0;
		for (int i = k + 1;i < N;i++) {
			float tmp2[4] = { m[i][k] , m[i][k] , m[i][k] , m[i][k] };
			__m128 tmp_ik = _mm_loadu_ps(tmp2);
			int num2 = k + 1;
			for (int j = k + 1;j + 4 <= N;j += 4, num2 = j) {
				__m128 tmp_ij = _mm_loadu_ps(m[i] + j);
				__m128 tmp_kj = _mm_loadu_ps(m[k] + j);
				tmp_kj = _mm_mul_ps(tmp_kj, tmp_ik);
				tmp_ij = _mm_sub_ps(tmp_ij, tmp_kj);
				_mm_storeu_ps(m[i] + j, tmp_ij);
			}
			for (int j = num2;j < N;j++) {
				m[i][j] -= m[k][j] * m[i][k];
			}
			m[i][k] = 0;
		}
	}
}
//对SSE两个部分进行并行化(对齐)
void parallel_SSE_Align() {
	for (int k = 0;k < N;k++) {
		//1.处理不对齐部分
		int pre1 = 4 - (k + 1) % 4;
		for (int j = k + 1;j < k + 1 + pre1;j++) {
			m[k][j] /= m[k][k];
		}
		//2.处理对齐部分
		float tmp1[4] = { m[k][k] ,m[k][k] ,m[k][k] ,m[k][k] };
		__m128 tmp_kk = _mm_load_ps(tmp1);
		int num1 = pre1 + k + 1;
		for (int j = pre1 + k + 1;j + 4 <= N;j += 4, num1 = j) {
			__m128 tmp_kj = _mm_load_ps(m[k] + j);
			tmp_kj = _mm_div_ps(tmp_kj, tmp_kk);
			_mm_store_ps(m[k] + j, tmp_kj);
		}
		//3.处理剩余部分
		for (int j = num1;j < N;j++) {
			m[k][j] /= m[k][k];
		}
		m[k][k] = 1.0;
		for (int i = k + 1;i < N;i++) {
			//1.处理不对齐部分
			int pre2 = 4 - (k + 1) % 4;
			for (int j = k + 1;j < k + 1 + pre2;j++) {
				m[i][j] -= m[k][j] * m[i][k];
			}
			//2.处理对齐部分
			float tmp2[4] = { m[i][k] , m[i][k] , m[i][k] , m[i][k] };
			__m128 tmp_ik = _mm_load_ps(tmp2);
			int num2 = pre2 + k + 1;
			for (int j = pre2 + k + 1;j + 4 <= N;j += 4, num2 = j) {
				__m128 tmp_ij = _mm_load_ps(m[i] + j);
				__m128 tmp_kj = _mm_load_ps(m[k] + j);
				tmp_kj = _mm_mul_ps(tmp_kj, tmp_ik);
				tmp_ij = _mm_sub_ps(tmp_ij, tmp_kj);
				_mm_store_ps(m[i] + j, tmp_ij);
			}
			//3.处理剩余部分
			for (int j = num2;j < N;j++) {
				m[i][j] -= m[k][j] * m[i][k];
			}
			m[i][k] = 0;
		}
	}
}
//对AVX两个部分进行并行化(不对齐)
void parallel_AVX_Unalign() {
	for (int k = 0;k < N;k++) {
		float tmp1[8] = { m[k][k] ,m[k][k] ,m[k][k] ,m[k][k],m[k][k],m[k][k],m[k][k],m[k][k] };
		__m256 tmp_kk = _mm256_loadu_ps(tmp1);
		int num1 = k + 1;
		for (int j = k + 1;j + 8 <= N;j += 8, num1 = j) {
			__m256 tmp_kj = _mm256_loadu_ps(m[k] + j);
			tmp_kj = _mm256_div_ps(tmp_kj, tmp_kk);
			_mm256_storeu_ps(m[k] + j, tmp_kj);
		}
		for (int j = num1;j < N;j++) {
			m[k][j] /= m[k][k];
		}
		m[k][k] = 1.0;
		for (int i = k + 1;i < N;i++) {
			float tmp2[8] = { m[i][k] , m[i][k] , m[i][k] , m[i][k], m[i][k] , m[i][k] , m[i][k] , m[i][k] };
			__m256 tmp_ik = _mm256_loadu_ps(tmp2);
			int num2 = k + 1;
			for (int j = k + 1;j + 8 <= N;j += 8, num2 = j) {
				__m256 tmp_ij = _mm256_loadu_ps(m[i] + j);
				__m256 tmp_kj = _mm256_loadu_ps(m[k] + j);
				tmp_kj = _mm256_mul_ps(tmp_kj, tmp_ik);
				tmp_ij = _mm256_sub_ps(tmp_ij, tmp_kj);
				_mm256_storeu_ps(m[i] + j, tmp_ij);
			}
			for (int j = num2;j < N;j++) {
				m[i][j] -= m[k][j] * m[i][k];
			}
			m[i][k] = 0;
		}
	}
}
//对AVX两个部分进行并行化(对齐)
void parallel_AVX_Align() {
	for (int k = 0;k < N;k++) {
		//1.处理不对齐部分
		int pre1 = 8 - (k + 1) % 8;
		for (int j = k + 1;j < k + 1 + pre1;j++) {
			m[k][j] /= m[k][k];
		}
		//2.处理对齐部分
		float tmp1[8] = { m[k][k] ,m[k][k] ,m[k][k] ,m[k][k],m[k][k],m[k][k],m[k][k],m[k][k] };
		__m256 tmp_kk = _mm256_load_ps(tmp1);
		int num1 = pre1 + k + 1;
		for (int j = k + 1 + pre1;j + 8 <= N;j += 8, num1 = j) {
			__m256 tmp_kj = _mm256_load_ps(m[k] + j);
			tmp_kj = _mm256_div_ps(tmp_kj, tmp_kk);
			_mm256_store_ps(m[k] + j, tmp_kj);
		}
		//3.处理剩余部分
		for (int j = num1;j < N;j++) {
			m[k][j] /= m[k][k];
		}
		m[k][k] = 1.0;
		for (int i = k + 1;i < N;i++) {
			//1.处理不对齐部分
			int pre2 = 8 - (k + 1) % 8;
			for (int j = k + 1;j < k + 1 + pre2;j++) {
				m[i][j] -= m[k][j] * m[i][k];
			}
			//2.处理对齐部分
			float tmp2[8] = { m[i][k] , m[i][k] , m[i][k] , m[i][k], m[i][k] , m[i][k] , m[i][k] , m[i][k] };
			__m256 tmp_ik = _mm256_load_ps(tmp2);
			int num2 = pre2 + k + 1;
			for (int j = pre2 + k + 1;j + 8 <= N;j += 8, num2 = j) {
				__m256 tmp_ij = _mm256_load_ps(m[i] + j);
				__m256 tmp_kj = _mm256_load_ps(m[k] + j);
				tmp_kj = _mm256_mul_ps(tmp_kj, tmp_ik);
				tmp_ij = _mm256_sub_ps(tmp_ij, tmp_kj);
				_mm256_store_ps(m[i] + j, tmp_ij);
			}
			//3.处理剩余部分
			for (int j = num2;j < N;j++) {
				m[i][j] -= m[k][j] * m[i][k];
			}
			m[i][k] = 0;
		}
	}
}
int main()
{
	long long begin, end, freq;
	double timeuse1 = 0, timeuse2 = 0, timeuse3 = 0, timeuse4 = 0, timeuse5 = 0;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	//对串行算法进行时间测试
	for (int i = 0;i < times;i++) {
		m_reset();
		QueryPerformanceCounter((LARGE_INTEGER*)&begin);
		serial_LU();
		QueryPerformanceCounter((LARGE_INTEGER*)&end);
		timeuse1 += (end - begin) * 1000.0 / freq;
	}
	cout << "n=" << N << " serial_LU:  " << timeuse1 / times << "ms" << endl;
	//对SSE两个部分（不对齐）并行化进行时间测试
	for (int i = 0;i < times;i++) {
		m_reset();
		QueryPerformanceCounter((LARGE_INTEGER*)&begin);
		parallel_SSE_Unalign();
		QueryPerformanceCounter((LARGE_INTEGER*)&end);
		timeuse2 += (end - begin) * 1000.0 / freq;
	}
	cout << "n=" << N << "  SSE_All_Part_Unalign:  " << timeuse2 / times << "ms" << endl;
	//对SSE两个部分（对齐）并行化进行时间测试
	for (int i = 0;i < times;i++) {
		m_reset();
		QueryPerformanceCounter((LARGE_INTEGER*)&begin);
		parallel_SSE_Align();
		QueryPerformanceCounter((LARGE_INTEGER*)&end);
		timeuse3 += (end - begin) * 1000.0 / freq;
	}
	cout << "n=" << N << "  SSE_All_Part_Align:  " << timeuse3 / times << "ms" << endl;
	//对AVX两个部分（不对齐）并行化进行时间测试
	for (int i = 0;i < times;i++) {
		m_reset();
		QueryPerformanceCounter((LARGE_INTEGER*)&begin);
		parallel_AVX_Unalign();
		QueryPerformanceCounter((LARGE_INTEGER*)&end);
		timeuse4 += (end - begin) * 1000.0 / freq;
	}
	cout << "n=" << N << "  AVX_All_Part_Unalign:  " << timeuse4 / times << "ms" << endl;
	//对AVX两个部分（对齐）并行化进行时间测试
	for (int i = 0;i < times;i++) {
		m_reset();
		QueryPerformanceCounter((LARGE_INTEGER*)&begin);
		parallel_AVX_Align();
		QueryPerformanceCounter((LARGE_INTEGER*)&end);
		timeuse5 += (end - begin) * 1000.0 / freq;
	}
	cout << "n=" << N << "  AVX_All_Part_Align:  " << timeuse5 / times << "ms" << endl;
	return 0;
}