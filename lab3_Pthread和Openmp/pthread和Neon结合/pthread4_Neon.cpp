#include<arm_neon.h>
#include<iostream>
#include<pthread.h>
#include<sys/time.h>
using namespace std;
const int N = 512;
float m[N][N];
const int thread_num = 6;
const int times = 10;
//�̲߳����ṹ��
struct Thread_Param {
	int t_id;
};
//barrier
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;
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
void* Thread_Func_Neon(void* param) {
	Thread_Param* p = (Thread_Param*)param;
	int t_id = p->t_id;
	for (int k = 0; k < N; k++)
	{
		//0�߳̽��г�������
		if (t_id == 0) {
			for (int j = k + 1; j < N; j++) {
				m[k][j] /= m[k][k];
			}
			m[k][k] = 1.0;
		}
		//��һ��ͬ����
		pthread_barrier_wait(&barrier_Divsion);
		//�����߳̽�����ȥ����
		for (int i = k + 1 + t_id; i < N; i += thread_num) {
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
		//�ڶ���ͬ����
		pthread_barrier_wait(&barrier_Elimination);
	}
	pthread_exit(NULL);
	return 0;
}
void Thread_main_Neon() {
	//��ʼ��barrier
	pthread_barrier_init(&barrier_Divsion, NULL, thread_num); //thread_num���̵߳����Ż�ִ��
	pthread_barrier_init(&barrier_Elimination, NULL, thread_num);
	//�����߳�
	pthread_t handles[thread_num];
	Thread_Param param[thread_num];
	for (int t_id = 0;t_id < thread_num;t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, Thread_Func_Neon, (void*)&param[t_id]);
	}
	//�ȴ������߳�
	for (int t_id = 0;t_id < thread_num;t_id++) {
		pthread_join(handles[t_id], NULL);
	}
	//����barrier
	pthread_barrier_destroy(&barrier_Divsion);
	pthread_barrier_destroy(&barrier_Elimination);
}
int main() {
	struct timeval begin, end;
	double timeuse = 0;
	for (int i = 0;i < times;i++) {
		m_reset();
		gettimeofday(&begin, NULL);
		Thread_main_Neon();
		gettimeofday(&end, NULL);
		timeuse += (end.tv_sec - begin.tv_sec) * 1000 + (double)(end.tv_usec - begin.tv_usec) / 1000.0;
	}
	cout << "n=" << N << " pthread4_Neon:  " << timeuse / times << "ms" << endl;

	return 0;
}