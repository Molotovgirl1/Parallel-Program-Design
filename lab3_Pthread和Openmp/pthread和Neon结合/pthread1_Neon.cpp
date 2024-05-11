#include<arm_neon.h>
#include<iostream>
#include<pthread.h>
#include<sys/time.h>
using namespace std;
const int N = 2048;
float m[N][N];
int thread_num = 6;
const int times = 10;
//线程参数结构体
struct Thread_Param
{
	int k; //消去的轮次
	int t_id; //线程id
};
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
void* Thread_Func_Neon(void* param) {
	Thread_Param* p = (Thread_Param*)param;
	int k = p->k;
	int t_id = p->t_id;
	for (int i = k + 1 + t_id;i < N;i += thread_num) {
		float tmp[4] = { m[i][k] ,m[i][k] ,m[i][k] ,m[i][k] };
		float32x4_t tmp_ik = vld1q_f32(tmp);
		int num = k + 1;
		for (int j = k + 1;j + 4 <= N;j += 4, num = j) {
			float32x4_t tmp_ij = vld1q_f32(m[i] + j);
			float32x4_t tmp_kj = vld1q_f32(m[k] + j);
			tmp_kj = vmulq_f32(tmp_kj, tmp_ik);
			tmp_ij = vsubq_f32(tmp_ij, tmp_kj);
			vst1q_f32(m[i] + j, tmp_ij);
		}
		for (int j = num;j < N;j++) {
			m[i][j] -= m[k][j] * m[i][k];
		}
		m[i][k] = 0;
	}
	pthread_exit(NULL);
	return 0;
}
void Thread_Main_Neon() {
	for (int k = 0;k < N;k++) {
		//主线程做除法操作
		for (int j = k + 1;j < N;j++) {
			m[k][j] /= m[k][k];
		}
		m[k][k] = 1.0;
		//创建子线程，进行消去操作
		pthread_t* handles = new pthread_t[thread_num]; //创建对应句柄
		Thread_Param* param = new Thread_Param[thread_num]; //创建对应参数
		//分配任务
		for (int t_id = 0;t_id < thread_num;t_id++) {
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//创建线程
		for (int t_id = 0;t_id < thread_num;t_id++) {
			pthread_create(&handles[t_id], NULL, Thread_Func_Neon, (void*)&param[t_id]);
		}
		//主线程等待回收所有子线程
		for (int t_id = 0;t_id < thread_num;t_id++) {
			pthread_join(handles[t_id], NULL);
		}
		//释放分配的空间
		delete[]handles;
		delete[]param;
	}
}
int main() {
	struct timeval begin, end;
	double timeuse = 0;
	for (int i = 0;i < times;i++) {
		m_reset();
		print();
		gettimeofday(&begin, NULL);
		Thread_Main_Neon();
		print();
		gettimeofday(&end, NULL);
		timeuse += (end.tv_sec - begin.tv_sec) * 1000 + (double)(end.tv_usec - begin.tv_usec) / 1000.0;
	}
	cout << "n=" << N << " pthread1_Neon:  " << timeuse / times << "ms" << endl;
	return 0;
}