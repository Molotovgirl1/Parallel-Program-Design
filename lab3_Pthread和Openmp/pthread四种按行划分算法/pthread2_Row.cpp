#include<iostream>
#include<pthread.h>
#include<semaphore.h>
#include<Windows.h>
#pragma comment(lib, "pthreadVC2.lib")
using namespace std;
const int N = 512;
float m[N][N];
const int thread_num = 6;
const int times = 100;
//线程参数结构体
struct Thread_Param {
	int t_id;
};
//信号量
sem_t sem_main;
sem_t sem_workerstart[thread_num];
sem_t sem_workerend[thread_num];
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
void* Thread_Func(void *param) {
	Thread_Param* p = (Thread_Param*)param;
	int t_id = p->t_id;
	for (int k = 0;k < N;k++) {
		sem_wait(&sem_workerstart[t_id]); //阻塞，等待主线程完成除法操作
		//循环划分任务
		for (int i = k + 1 + t_id;i < N;i += thread_num) {
			for (int j = k + 1;j < N;j++) {
				m[i][j] -= (m[i][k] * m[k][j]);
			}
			m[i][k] = 0;
		}
		sem_post(&sem_main); //唤醒主线程
		sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮
	}
	pthread_exit(NULL);
	return 0;
}
void Thread_main() {
	/*------初始化创建工作------*/
	sem_init(&sem_main, 0, 0);
	for (int t_id = 0;t_id < thread_num;t_id++) {
		sem_init(&sem_workerstart[t_id], 0, 0);
	}
	for (int t_id = 0;t_id < thread_num;t_id++) {
		sem_init(&sem_workerend[t_id], 0, 0);
	}
	pthread_t handles[thread_num]; //创建对应的句柄
	Thread_Param param[thread_num]; //创建对应的参数
	for (int t_id = 0;t_id < thread_num;t_id++) {
		param[t_id].t_id = t_id;
	}
	for (int t_id = 0;t_id < thread_num;t_id++) {
		pthread_create(&handles[t_id], NULL, Thread_Func, (void*)&param[t_id]);
	}
	/*------主线程工作-------*/
	for (int k = 0;k < N;k++) {
		//除法操作
		for (int j = k + 1;j < N;j++) {
			m[k][j] /= m[k][k];
		}
		m[k][k] = 1.0;
		//开始唤醒工作线程做消去操作
		for (int t_id = 0;t_id < thread_num;t_id++) {
			sem_post(&sem_workerstart[t_id]);
		}
		//主线程睡眠（等待工作线程完成此轮消去操作）
		for (int t_id = 0;t_id < thread_num;t_id++) {
			sem_wait(&sem_main);
		}
		//唤醒工作线程进入下一轮循环
		for (int t_id = 0;t_id < thread_num;t_id++) {
			sem_post(&sem_workerend[t_id]);
		}
	}
	/*---------结束回收工作---------*/
	//等待回收工作线程
	for (int t_id = 0;t_id < thread_num;t_id++) {
		pthread_join(handles[t_id], NULL);
	}
	//销毁信号量
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
	cout << "n=" << N << " pthread2 :  " << timeuse / times << "ms" << endl;

	return 0;
}
