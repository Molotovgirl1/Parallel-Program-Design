#include<iostream>
#include<pthread.h>
#include<semaphore.h>
#include<Windows.h>
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
sem_t sem_main;
sem_t sem_workerstart[thread_num];
sem_t sem_workerend[thread_num];
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
void* Thread_Func(void *param) {
	Thread_Param* p = (Thread_Param*)param;
	int t_id = p->t_id;
	for (int k = 0;k < N;k++) {
		sem_wait(&sem_workerstart[t_id]); //�������ȴ����߳���ɳ�������
		//ѭ����������
		for (int i = k + 1 + t_id;i < N;i += thread_num) {
			for (int j = k + 1;j < N;j++) {
				m[i][j] -= (m[i][k] * m[k][j]);
			}
			m[i][k] = 0;
		}
		sem_post(&sem_main); //�������߳�
		sem_wait(&sem_workerend[t_id]); //�������ȴ����̻߳��ѽ�����һ��
	}
	pthread_exit(NULL);
	return 0;
}
void Thread_main() {
	/*------��ʼ����������------*/
	sem_init(&sem_main, 0, 0);
	for (int t_id = 0;t_id < thread_num;t_id++) {
		sem_init(&sem_workerstart[t_id], 0, 0);
	}
	for (int t_id = 0;t_id < thread_num;t_id++) {
		sem_init(&sem_workerend[t_id], 0, 0);
	}
	pthread_t handles[thread_num]; //������Ӧ�ľ��
	Thread_Param param[thread_num]; //������Ӧ�Ĳ���
	for (int t_id = 0;t_id < thread_num;t_id++) {
		param[t_id].t_id = t_id;
	}
	for (int t_id = 0;t_id < thread_num;t_id++) {
		pthread_create(&handles[t_id], NULL, Thread_Func, (void*)&param[t_id]);
	}
	/*------���̹߳���-------*/
	for (int k = 0;k < N;k++) {
		//��������
		for (int j = k + 1;j < N;j++) {
			m[k][j] /= m[k][k];
		}
		m[k][k] = 1.0;
		//��ʼ���ѹ����߳�����ȥ����
		for (int t_id = 0;t_id < thread_num;t_id++) {
			sem_post(&sem_workerstart[t_id]);
		}
		//���߳�˯�ߣ��ȴ������߳���ɴ�����ȥ������
		for (int t_id = 0;t_id < thread_num;t_id++) {
			sem_wait(&sem_main);
		}
		//���ѹ����߳̽�����һ��ѭ��
		for (int t_id = 0;t_id < thread_num;t_id++) {
			sem_post(&sem_workerend[t_id]);
		}
	}
	/*---------�������չ���---------*/
	//�ȴ����չ����߳�
	for (int t_id = 0;t_id < thread_num;t_id++) {
		pthread_join(handles[t_id], NULL);
	}
	//�����ź���
	sem_destroy(&sem_main);
	for (int t_id = 0;t_id < thread_num;t_id++) {
		sem_destroy(&sem_workerstart[t_id]);
	}
	for (int t_id = 0;t_id < thread_num;t_id++) {
		sem_destroy(&sem_workerend[t_id]);
	}
}
int main() {
	long long begin, end, freq;
	double timeuse = 0;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	for (int i = 0;i < times;i++) {
		m_reset();
		QueryPerformanceCounter((LARGE_INTEGER*)&begin);
		Thread_main();
		QueryPerformanceCounter((LARGE_INTEGER*)&end);
		timeuse += (end - begin) * 1000.0 / freq;
	}
	cout << "n=" << N << " Static threads + semaphores :  " << timeuse / times << "ms" << endl;

	return 0;
}
