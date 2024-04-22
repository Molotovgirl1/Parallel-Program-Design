#include<arm_neon.h>
#include<iostream>
#include<sys/time.h>
#include<stdlib.h>
using namespace std;
const int N = 256;
float m[N][N];
const int times = 10;
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
//��ͨ��˹��Ԫ�����㷨
void serial_LU() {
	for (int k = 0;k < N;k++) {
		for (int j = k + 1;j < N;j++) {
			m[k][j] /= m[k][k]; //����k�жԽ���Ԫ�ر�Ϊ1
		}
		m[k][k] = 1.0;
		for (int i = k + 1;i < N;i++) {
			for (int j = k + 1;j < N;j++) {
				m[i][j] -= m[k][j] * m[i][k]; //�ӵ�i����ȥ��k��
			}
			m[i][k] = 0;
		}
	}
}
//��һ���ֽ���Neon���л�
void parallel_Neon_Part1() {
	for (int k = 0;k < N;k++) {
		float tmp[4] = { m[k][k],m[k][k] ,m[k][k] ,m[k][k] };
		float32x4_t tmp_vec = vld1q_f32(tmp);
		int num = k + 1;
		for (int j = k + 1;j + 4 <= N;j += 4,num=j) {
			float32x4_t m_vec = vld1q_f32(m[k] + j);
			m_vec = vdivq_f32(m_vec, tmp_vec);
			vst1q_f32(m[k] + j, m_vec);
		}
		for (int j = num ;j < N;j++) {
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

//�Եڶ����ֽ���Neon���л�
void parallel_Neon_Part2() {
	for (int k = 0;k < N;k++) {
		for (int j = k + 1;j < N;j++) {
			m[k][j] /= m[k][k];
		}
		m[k][k] = 1.0;
		for (int i = k + 1;i < N;i++) {
			float tmp[4] = { m[i][k] ,m[i][k] ,m[i][k] ,m[i][k] };
			float32x4_t tmp_ik= vld1q_f32(tmp);
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
	}
}
//��������ͬʱ����Neon���л�
void parallel_Neon() {
	for (int k = 0;k < N;k++) {
		float tmp1[4] = { m[k][k] , m[k][k] ,m[k][k] ,m[k][k] };
		float32x4_t tmp_kk = vld1q_f32(tmp1);
		int num1 = k + 1;
		for (int j = k + 1;j + 4 <= N;j += 4, num1 = j) {
			float32x4_t tmp_kj = vld1q_f32(m[k] + j);
			tmp_kj = vdivq_f32(tmp_kj, tmp_kk);
			vst1q_f32(m[k] + j, tmp_kj);
		}
		for (int j = num1;j < N;j++) {
			m[k][j] /= m[k][k];
		}
		m[k][k] = 1.0;
		for (int i = k + 1;i < N;i++) {
			float tmp2[4] = { m[i][k] ,m[i][k] ,m[i][k] ,m[i][k] };
			float32x4_t tmp_ik = vld1q_f32(tmp2);
			int num2 = k + 1;
			for (int j = k + 1;j + 4 <= N;j += 4, num2 = j) {
				float32x4_t tmp_ij = vld1q_f32(m[i] + j);
				float32x4_t tmp_kj = vld1q_f32(m[k] + j);
				tmp_kj = vmulq_f32(tmp_kj, tmp_ik);
				tmp_ij = vsubq_f32(tmp_ij, tmp_kj);
				vst1q_f32(m[i] + j, tmp_ij);
			}
			for (int j = num2;j < N;j++) {
				m[i][j] -= m[k][j] * m[i][k];
			}
			m[i][k] = 0;
		}
	}
}
int main() {
	struct timeval begin, end;
	double timeuse1 = 0, timeuse2 = 0, timeuse3 = 0, timeuse4 = 0;
	//�Դ����㷨����ʱ�����
	for (int i = 0;i < times;i++) {
		m_reset();
		gettimeofday(&begin, NULL);
		serial_LU();
		gettimeofday(&end, NULL);
		timeuse1 += (end.tv_sec - begin.tv_sec) * 1000 + (double)(end.tv_usec - begin.tv_usec) / 1000.0;
	}
	cout << "n=" << N << " serial_LU:  " << timeuse1 / times << "ms" << endl;
	//�Ե�һ����Neon���л�����ʱ�����
	for (int i = 0;i < times;i++) {
		m_reset();
		gettimeofday(&begin, NULL);
		parallel_Neon_Part1();
		gettimeofday(&end, NULL);
		timeuse2 += (end.tv_sec - begin.tv_sec) * 1000 + (double)(end.tv_usec - begin.tv_usec) / 1000.0;
	}
	cout << "n=" << N << "  Part1:  " << timeuse2 / times << "ms" << endl;
	//�Եڶ�����Neon���л�����ʱ�����
	for (int i = 0;i < times;i++) {
		m_reset();
		gettimeofday(&begin, NULL);
		parallel_Neon_Part2();
		gettimeofday(&end, NULL);
		timeuse3 += (end.tv_sec - begin.tv_sec) * 1000 + (double)(end.tv_usec - begin.tv_usec) / 1000.0;
	}
	cout << "n=" << N << "  Part2:  " << timeuse3 / times << "ms" << endl;
	//��������Neon���л�����ʱ�����
	for (int i = 0;i < times;i++) {
		m_reset();
		gettimeofday(&begin, NULL);
		parallel_Neon();
		gettimeofday(&end, NULL);
		timeuse4 += (end.tv_sec - begin.tv_sec) * 1000 + (double)(end.tv_usec - begin.tv_usec) / 1000.0;
	}
	cout << "n=" << N << "  All_Part:  " << timeuse4 / times << "ms" << endl;
	return 0;
}
