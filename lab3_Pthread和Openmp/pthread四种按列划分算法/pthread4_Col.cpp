#include<iostream>
#include<pthread.h>
#include<Windows.h>
#pragma comment(lib, "pthreadVC2.lib")
using namespace std;
const int N = 512;
float m[N][N];
const int thread_num = 6;
const int times = 10;
//线程参数结构体
struct Thread_Param {
	int t_id;
};
//barrier
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;
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
void* Thread_Func_Col(void* param) {
	Thread_Param* p = (Thread_Param*)param;
	int t_id = p->t_id;
	for (int k = 0; k < N; k++)
	{
		//0线程进行除法操作
		if (t_id == 0) {
			for (int j = k + 1; j < N; j++) {
				m[k][j] /= m[k][k];
			}
			m[k][k] = 1.0;
		}
		//第一个同步点
		pthread_barrier_wait(&barrier_Divsion);
		//所有线程进行消去操作
		for (int i = k + 1 ; i < N; i++) {
			for (int j = k + 1+t_id; j < N; j+=thread_num) {
				m[i][j] -= (m[i][k] * m[k][j]);
			}
		}
		//第二个同步点
		pthread_barrier_wait(&barrier_Elimination);
		//将m[i][k]置为0
		for (int i = k + 1;i < N;i++) {
			m[i][k] = 0;
		}
	}
	pthread_exit(NULL);
	return 0;
}
void Thread_main_Col() {
	//初始化barrier
	pthread_barrier_init(&barrier_Divsion, NULL, thread_num); //thread_num个线程到达后才会执行
	pthread_barrier_init(&barrier_Elimination, NULL, thread_num);
	//创建线程
	pthread_t handles[thread_num];
	Thread_Param param[thread_num];
	for (int t_id = 0;t_id < thread_num;t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, Thread_Func_Col, (void*)&param[t_id]);
	}
	//等待回收线程
	for (int t_id = 0;t_id < thread_num;t_id++) {
		pthread_join(handles[t_id], NULL);
	}
	//销毁barrier
	pthread_barrier_destroy(&barrier_Divsion);
	pthread_barrier_destroy(&barrier_Elimination);
}
int main() {
	long long begin, end, freq;
	double timeuse = 0;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	for (int i = 0;i < times;i++) {
		m_reset();
		QueryPerformanceCounter((LARGE_INTEGER*)&begin);
		Thread_main_Col();
		QueryPerformanceCounter((LARGE_INTEGER*)&end);
		timeuse += (end - begin) * 1000.0 / freq;
	}
	cout << "n=" << N << " pthread4_Col:  " << timeuse / times << "ms" << endl;

	return 0;
}