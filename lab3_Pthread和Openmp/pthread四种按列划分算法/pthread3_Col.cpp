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
//线程参数结构体
struct Thread_Param {
	int t_id;
};
//信号量
sem_t sem_leader;
sem_t sem_Divsion[thread_num - 1];
sem_t sem_Elimination[thread_num - 1];
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
	for (int k = 0;k < N;k++) {
		//0线程进行除法操作
		if (t_id == 0) {
			for (int j = k + 1; j < N; j++) {
				m[k][j] /= m[k][k];
			}
			m[k][k] = 1.0;
		}
		else {
			sem_wait(&sem_Divsion[t_id - 1]); //阻塞，等待完成除法操作
		}
		//唤醒其他工作线程进行消去操作
		if (t_id == 0) {
			for (int i = 0;i < thread_num - 1;i++) {
				sem_post(&sem_Divsion[i]);
			}
		}
		//循环划分任务给工作线程（包括0线程）
		for (int i = k + 1 ;i < N;i++) {
			for (int j = k + 1 + t_id; j < N; j += thread_num)
				m[i][j] -= (m[i][k] * m[k][j]);
		}
		//等待进入下一轮循环
		if (t_id == 0) {
			for (int i = 0;i < thread_num - 1;i++) {
				sem_wait(&sem_leader); //阻塞0号线程，等待其他工作线程完成消去
			}
			for (int i = 0;i < thread_num - 1;i++) {
				sem_post(&sem_Elimination[i]); //唤醒其他工作线程进入下一轮
			}
		}
		else {
			sem_post(&sem_leader); //唤醒0号线程，表示已完成消去任务
			sem_wait(&sem_Elimination[t_id - 1]); //阻塞，等待通知进入下一轮
		}
		//将m[i][k]置为0
		for (int i = k + 1;i < N;i++) {
			m[i][k] = 0;
		}
	}
	pthread_exit(NULL);
	return 0;
}
void Thread_main_Col() {
	/*--------初始化创建工作--------*/
	//初始化信号量
	sem_init(&sem_leader, 0, 0);
	for (int i = 0;i < thread_num - 1;i++) {
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}
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
	//销毁信号量
	sem_destroy(&sem_leader);
	for (int i = 0;i < thread_num - 1;i++) {
		sem_destroy(&sem_Divsion[i]);
		sem_destroy(&sem_Elimination[i]);
	}
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
	cout << "n=" << N << " pthread3_Col: " << timeuse / times << "ms" << endl;

	return 0;
}