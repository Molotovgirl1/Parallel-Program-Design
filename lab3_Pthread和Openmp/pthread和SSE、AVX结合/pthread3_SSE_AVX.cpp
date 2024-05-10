#include<iostream>
#include<pthread.h>
#include<semaphore.h>
#include<Windows.h>
#include<nmmintrin.h> //SSSE4.2
#include<immintrin.h> //AVX��AVX
#pragma comment(lib, "pthreadVC2.lib")
using namespace std;
const int N = 512;
float m[N][N];
const int thread_num = 6;
const int times = 10;
//�̲߳����ṹ��
struct Thread_Param {
	int t_id;
};
//�ź���
sem_t sem_leader;
sem_t sem_Divsion[thread_num - 1];
sem_t sem_Elimination[thread_num - 1];
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
	for (int k = 0;k < N;k++) {
		for (int i = k + 1;i < N;i++) {
			for (int j = 0;j < N;j++) {
				m[i][j] += m[k][j];
			}
		}
	}
}
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
void* Thread_Func_SSE(void* param) {
	Thread_Param* p = (Thread_Param*)param;
	int t_id = p->t_id;
	for (int k = 0;k < N;k++) {
		//0�߳̽��г�������
		if (t_id == 0) {
			for (int j = k + 1; j < N; j++) {
				m[k][j] /= m[k][k];
			}
			m[k][k] = 1.0;
		}
		else {
			sem_wait(&sem_Divsion[t_id - 1]); //�������ȴ���ɳ�������
		}
		//�������������߳̽�����ȥ����
		if (t_id == 0) {
			for (int i = 0;i < thread_num - 1;i++) {
				sem_post(&sem_Divsion[i]);
			}
		}
		//ѭ����������������̣߳�����0�̣߳�
		for (int i = k + 1 + t_id;i < N;i += thread_num) {
			float tmp[4] = { m[i][k] , m[i][k] , m[i][k] , m[i][k] };
			__m128 tmp_ik = _mm_loadu_ps(tmp);
			int num = k + 1;
			for (int j = k + 1;j + 4 <= N;j += 4, num = j) {
				__m128 tmp_ij = _mm_loadu_ps(m[i] + j);
				__m128 tmp_kj = _mm_loadu_ps(m[k] + j);
				tmp_kj = _mm_mul_ps(tmp_kj, tmp_ik);
				tmp_ij = _mm_sub_ps(tmp_ij, tmp_kj);
				_mm_storeu_ps(m[i] + j, tmp_ij);
			}
			for (int j = num;j < N;j++) {
				m[i][j] -= m[k][j] * m[i][k];
			}
			m[i][k] = 0;
		}
		//�ȴ�������һ��ѭ��
		if (t_id == 0) {
			for (int i = 0;i < thread_num - 1;i++) {
				sem_wait(&sem_leader); //����0���̣߳��ȴ����������߳������ȥ
			}
			for (int i = 0;i < thread_num - 1;i++) {
				sem_post(&sem_Elimination[i]); //�������������߳̽�����һ��
			}
		}
		else {
			sem_post(&sem_leader); //����0���̣߳���ʾ�������ȥ����
			sem_wait(&sem_Elimination[t_id - 1]); //�������ȴ�֪ͨ������һ��
		}
	}
	pthread_exit(NULL);
	return 0;
}
void Thread_main_SSE() {
	/*--------��ʼ����������--------*/
	//��ʼ���ź���
	sem_init(&sem_leader, 0, 0);
	for (int i = 0;i < thread_num - 1;i++) {
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}
	//�����߳�
	pthread_t handles[thread_num];
	Thread_Param param[thread_num];
	for (int t_id = 0;t_id < thread_num;t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, Thread_Func_SSE, (void*)&param[t_id]);
	}
	//�ȴ������߳�
	for (int t_id = 0;t_id < thread_num;t_id++) {
		pthread_join(handles[t_id], NULL);
	}
	//�����ź���
	sem_destroy(&sem_leader);
	for (int i = 0;i < thread_num - 1;i++) {
		sem_destroy(&sem_Divsion[i]);
		sem_destroy(&sem_Elimination[i]);
	}
}
void* Thread_Func_AVX(void* param) {
	Thread_Param* p = (Thread_Param*)param;
	int t_id = p->t_id;
	for (int k = 0;k < N;k++) {
		//0�߳̽��г�������
		if (t_id == 0) {
			for (int j = k + 1; j < N; j++) {
				m[k][j] /= m[k][k];
			}
			m[k][k] = 1.0;
		}
		else {
			sem_wait(&sem_Divsion[t_id - 1]); //�������ȴ���ɳ�������
		}
		//�������������߳̽�����ȥ����
		if (t_id == 0) {
			for (int i = 0;i < thread_num - 1;i++) {
				sem_post(&sem_Divsion[i]);
			}
		}
		//ѭ����������������̣߳�����0�̣߳�
		for (int i = k + 1 + t_id;i < N;i += thread_num) {
			float tmp[8] = { m[i][k] , m[i][k] , m[i][k] , m[i][k],m[i][k],m[i][k],m[i][k],m[i][k] };
			__m256 tmp_ik = _mm256_loadu_ps(tmp);
			int num = k + 1;
			for (int j = k + 1;j + 8 <= N;j += 8, num = j) {
				__m256 tmp_ij = _mm256_loadu_ps(m[i] + j);
				__m256 tmp_kj = _mm256_loadu_ps(m[k] + j);
				tmp_kj = _mm256_mul_ps(tmp_kj, tmp_ik);
				tmp_ij = _mm256_sub_ps(tmp_ij, tmp_kj);
				_mm256_storeu_ps(m[i] + j, tmp_ij);
			}
			for (int j = num;j < N;j++) {
				m[i][j] -= m[k][j] * m[i][k];
			}
			m[i][k] = 0;
		}
		//�ȴ�������һ��ѭ��
		if (t_id == 0) {
			for (int i = 0;i < thread_num - 1;i++) {
				sem_wait(&sem_leader); //����0���̣߳��ȴ����������߳������ȥ
			}
			for (int i = 0;i < thread_num - 1;i++) {
				sem_post(&sem_Elimination[i]); //�������������߳̽�����һ��
			}
		}
		else {
			sem_post(&sem_leader); //����0���̣߳���ʾ�������ȥ����
			sem_wait(&sem_Elimination[t_id - 1]); //�������ȴ�֪ͨ������һ��
		}
	}
	pthread_exit(NULL);
	return 0;
}
void Thread_main_AVX() {
	/*--------��ʼ����������--------*/
	//��ʼ���ź���
	sem_init(&sem_leader, 0, 0);
	for (int i = 0;i < thread_num - 1;i++) {
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}
	//�����߳�
	pthread_t handles[thread_num];
	Thread_Param param[thread_num];
	for (int t_id = 0;t_id < thread_num;t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, Thread_Func_AVX, (void*)&param[t_id]);
	}
	//�ȴ������߳�
	for (int t_id = 0;t_id < thread_num;t_id++) {
		pthread_join(handles[t_id], NULL);
	}
	//�����ź���
	sem_destroy(&sem_leader);
	for (int i = 0;i < thread_num - 1;i++) {
		sem_destroy(&sem_Divsion[i]);
		sem_destroy(&sem_Elimination[i]);
	}
}
int main() {
	long long begin, end, freq;
	double timeuse1 = 0, timeuse2 = 0;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	//SSE
	for (int i = 0;i < times;i++) {
		m_reset();
		QueryPerformanceCounter((LARGE_INTEGER*)&begin);
		Thread_main_SSE();
		QueryPerformanceCounter((LARGE_INTEGER*)&end);
		timeuse1 += (end - begin) * 1000.0 / freq;
	}
	cout << "n=" << N << " pthread3_SSE :  " << timeuse1 / times << "ms" << endl;
	//AVX
	for (int i = 0;i < times;i++) {
		m_reset();
		QueryPerformanceCounter((LARGE_INTEGER*)&begin);
		Thread_main_AVX();
		QueryPerformanceCounter((LARGE_INTEGER*)&end);
		timeuse2 += (end - begin) * 1000.0 / freq;
	}
	cout << "n=" << N << " pthread3_AVX :  " << timeuse2 / times << "ms" << endl;
	return 0;
}