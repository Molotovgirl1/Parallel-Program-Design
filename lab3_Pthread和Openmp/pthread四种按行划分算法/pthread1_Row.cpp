#include<iostream>
#include<pthread.h>
#include<Windows.h>
#pragma comment(lib, "pthreadVC2.lib")
using namespace std;
const int N = 512;
float m[N][N];
int thread_num = 6;
const int times = 100;
//�̲߳����ṹ��
struct Thread_Param
{
	int k; //��ȥ���ִ�
	int t_id; //�߳�id
};
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
void *Thread_Func(void *param) {
	Thread_Param* p = (Thread_Param*)param;
	int k = p->k;
	int t_id = p->t_id;
	for (int i = k + 1 + t_id;i < N;i+=thread_num) {
		for (int j = k + 1;j < N;j++) {
			m[i][j] -= (m[i][k] * m[k][j]);
		}
		m[i][k] = 0;
	}
	pthread_exit(NULL);
	return 0;
}
void Thread_Main() {
	for (int k = 0;k < N;k++) {
		//���߳�����������
		for (int j = k + 1;j < N;j++) {
			m[k][j] /= m[k][k];
		}
		m[k][k] = 1.0;
		//�������̣߳�������ȥ����
		pthread_t* handles = new pthread_t[thread_num]; //������Ӧ���
		Thread_Param* param = new Thread_Param[thread_num]; //������Ӧ����
		//��������
		for (int t_id = 0;t_id < thread_num;t_id++) {
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//�����߳�
		for (int t_id = 0;t_id < thread_num;t_id++) {
			pthread_create(&handles[t_id], NULL, Thread_Func, (void*)&param[t_id]);
		}
		//���̵߳ȴ������������߳�
		for (int t_id = 0;t_id < thread_num;t_id++) {
			pthread_join(handles[t_id], NULL);
		}
		//�ͷŷ���Ŀռ�
		delete[]handles;
		delete[]param;
	}
}
int main() {
	long long begin, end, freq;
	double timeuse = 0;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	for (int i = 0;i < times;i++) {
		m_reset();
		QueryPerformanceCounter((LARGE_INTEGER*)&begin);
		Thread_Main();
		QueryPerformanceCounter((LARGE_INTEGER*)&end);
		timeuse+= (end - begin) * 1000.0 / freq;
	}
	cout << "n=" << N << " pthread1:  " << timeuse / times << "ms" << endl;
	return 0;
}