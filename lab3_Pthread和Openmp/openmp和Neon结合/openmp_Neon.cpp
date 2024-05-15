#include <arm_neon.h>
#include<iostream>
#include<sys/time.h>
#include<omp.h>
using namespace std;
const int N = 5;
float m[N][N];
const int NUM_THREADS = 6;
const int times = 3;
//输出数组
void print()
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++) {
			cout << m[i][j] << " ";
		}
		cout << endl;
	}
}
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
	print();
	for (int k = 0;k < N;k++) {
		for (int i = k + 1;i < N;i++) {
			for (int j = 0;j < N;j++) {
				m[i][j] += m[k][j];
			}
		}
	}
}
void Openmp_Neon() {
	int i, j, k;
	float tmp;
	//创建线程
#pragma omp parallel num_threads(NUM_THREADS),private(i,j,k,tmp)
	for (k = 0;k < N;k++) {
		//除法部分，一个线程执行
#pragma omp single 
		{
			tmp = m[k][k];
			for (j = k + 1;j < N;j++) {
				m[k][j] /= tmp;
			}
			m[k][k] = 1.0;
		}
		//消去部分，使用行划分
#pragma omp for
		for (i = k + 1;i < N;i++) {
			float temp[4] = { m[i][k] ,m[i][k] ,m[i][k] ,m[i][k] };
			float32x4_t tmp_ik = vld1q_f32(temp);
			int num = k + 1;
			for (j = k + 1;j + 4 <= N;j += 4, num = j) {
				float32x4_t tmp_ij = vld1q_f32(m[i] + j);
				float32x4_t tmp_kj = vld1q_f32(m[k] + j);
				tmp_kj = vmulq_f32(tmp_kj, tmp_ik);
				tmp_ij = vsubq_f32(tmp_ij, tmp_kj);
				vst1q_f32(m[i] + j, tmp_ij);
			}
			for (j = num;j < N;j++) {
				m[i][j] -= m[k][j] * m[i][k];
			}
			m[i][k] = 0;
		}
	}
}
int main() {
	struct timeval begin, end;
	double timeuse = 0;
	for (int i = 0;i < times;i++) {
		m_reset();
		gettimeofday(&begin, NULL);
		Openmp_Neon();
		print();
		gettimeofday(&end, NULL);
		timeuse += (end.tv_sec - begin.tv_sec) * 1000 + (double)(end.tv_usec - begin.tv_usec) / 1000.0;
	}
	cout << "n=" << N << " OPenMP_Neon:  " << timeuse / times << "ms" << endl;
	return 0;
}