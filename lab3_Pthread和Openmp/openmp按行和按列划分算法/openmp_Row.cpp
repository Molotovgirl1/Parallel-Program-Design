#include<iostream>
#include<Windows.h>
#include<omp.h>
using namespace std;
const int N = 512;
float m[N][N];
const int NUM_THREADS = 6;
const int times = 10;
//�������
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
// ���ɲ�������
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
	//print();
	for (int k = 0;k < N;k++) {
		for (int i = k + 1;i < N;i++) {
			for (int j = 0;j < N;j++) {
				m[i][j] += m[k][j];
			}
		}
	}
}
//��ͨ��˹��Ԫ�㷨
void serial_LU() {
	for (int k = 0;k < N;k++) {
		for (int j = k + 1;j < N;j++) {
			m[k][j] /= m[k][k]; 
		}
		m[k][k] = 1.0;
		for (int i = k + 1;i < N;i++) {
			for (int j = k + 1;j < N;j++) {
				m[i][j] -= m[k][j] * m[i][k];
			}
			m[i][k] = 0;
		}
	}
}
void Openmp_Row() {
	int i, j, k;
	int tmp;
	//�����߳�
#pragma omp parallel num_thread(NUM_THREADS),private(i,j,j,tmp)
	for (k = 0;k < N;k++) {
		//�������֣�һ���߳�ִ��
#pragma omp single 
		{
			tmp = m[k][k];
			for (j = k + 1;j < N;j++) {
				m[k][j] /= tmp;
			}
			m[k][k] = 1.0;
		}
		//��ȥ���֣�ʹ���л���
#pragma omp for
		for (i = k + 1;i < N;i++) {
			tmp = m[i][k];
			for (int j = k + 1;j < N;j++) {
				m[i][j] -= (tmp * m[k][j]);
			}
			m[i][k] = 0;
		}
	}
}
int main() {
	long long begin, end, freq;
	double timeuse1 = 0, timeuse2 = 0;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	//��ͨ��˹��Ԫ��ʱ
	for (int i = 0;i < times;i++) {
		m_reset();
		QueryPerformanceCounter((LARGE_INTEGER*)&begin);
		serial_LU();
		//print();
		QueryPerformanceCounter((LARGE_INTEGER*)&end);
		timeuse1 += (end - begin) * 1000.0 / freq;
	}
	cout << "n=" << N << " Serial:  " << timeuse1 / times << "ms" << endl;
	//OpenMP���л��ּ�ʱ
	for (int i = 0;i < times;i++) {
		m_reset();
		QueryPerformanceCounter((LARGE_INTEGER*)&begin);
		Openmp_Row();
		//print();
		QueryPerformanceCounter((LARGE_INTEGER*)&end);
		timeuse2 += (end - begin) * 1000.0 / freq;
	}
	cout << "n=" << N << " OPenMP_Row:  " << timeuse2 / times << "ms" << endl;
	return 0;
}
