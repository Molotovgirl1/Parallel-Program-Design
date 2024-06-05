#include<iostream>
//#include<stdlib.h>
#include<Windows.h>
#include<mpi.h>
using namespace std;
const int N = 256;
float m[N][N];
const int times = 10;
double time1 = 0, time2 = 0;
void print() {
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
//MPI:�黮�ַ�ʽ(����ȫ�������һ�����̣�
void MPI_block() {
	double Tstart, Tend;
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); //����ʶ����ý��̵�rank
	MPI_Comm_size(MPI_COMM_WORLD, &size); //���������
	Tstart = MPI_Wtime();
	int tasks = floor(N * 1.0 / size); //ÿ�����̵���������
	//0�Ž��̷ַ�����
	if (rank == 0) {
		for (int i = 1;i < size;i++) {
			int pos = i * tasks;
			if (i != size - 1) {
				MPI_Send(&m[pos][0], tasks * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
			}
			else
			{
				MPI_Send(&m[pos][0], (N - pos) * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD); //���һ���������ⷢ��
			}
		}
	}
	//�������̽�������
	else
	{
		if (rank != size - 1) {
			MPI_Recv(&m[rank * tasks][0], tasks * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
		}
		else {
			MPI_Recv(&m[rank * tasks][0], (N - rank * tasks) * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE); //���һ�������������
		}
	}
	//��ʼ����
	int start = rank * tasks; //��ʼ��
	int end; //������
	if (rank != size - 1) {
		end = (rank + 1) * tasks;
	}
	else
	{
		end = N;
	}
	for (int k = 0;k < N;k++) {
		//�������㲢�㲥
		if (k >= start && k < end) {
			for (int j = k + 1;j < N;j++) {
				m[k][j] /= m[k][k];
			}
			m[k][k] = 1.0;
			for (int i = 0;i < size;i++) {
				if (i != rank) {
					MPI_Send(&m[k][0], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
				}
			}
		}
		//������̽��ճ������
		else {
			MPI_Recv(&m[k][0], N, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		//��Ԫ����
		for (int i = max(k + 1, start);i < end;i++) {
			for (int j = k + 1;j < N;j++) {
				m[i][j] = m[i][j] - m[i][k] * m[k][j];
			}
			m[i][k] = 0;
		}
	}
	//������ʱ
	Tend = MPI_Wtime();
	if (rank == 0)
	{
		//print();
		time1 += (Tend - Tstart) * 1000.0;
	}
	return;
}
//MPI:ѭ������
void MPI_cycle() {
	double Tstart, Tend;
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	Tstart = MPI_Wtime();
	int tasks; //ÿ������������
	if (rank < N % size) {
		tasks = N / size + 1;
	}
	else
	{
		tasks = N / size;
	}
	//0�Ž��зַ�����
	float* buff = new float[tasks * N]; //�ݴ�Ҫ�ַ���ĳ�����̵�����
	if (rank == 0) {
		for (int p = 1;p < size;p++) {
			for (int i = p;i < N;i += size) { //��sizeΪ�����������ݵ�buff
				for (int j = 0;j < N;j++) {
					buff[i / size * N + j] = m[i][j];
				}
			}
			int ptasks = p < N % size ? N / size + 1 : N / size;//�����ս��̵���������
			MPI_Send(buff, ptasks * N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
		}
	}
	//��0���̽�������
	else {
		MPI_Recv(&m[rank][0], tasks * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		//�����յ������ݵ�������Ӧλ��
		for (int i = 0;i < tasks;i++) {
			for (int j = 0;j < N;j++) {
				m[rank + i * size][j] = m[rank + i][j];
			}
		}
	}
	//��Ԫ����
	for (int k = 0;k < N;k++) {
		//����������
		if (k % size == rank) {
			for (int j = k + 1;j < N;j++) {
				m[k][j] /= m[k][k];
			}
			m[k][k] = 1.0;
			//�㲥���
			for (int i = 0;i < size;i++) {
				if (i != rank) {
					MPI_Send(&m[k][0], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
				}
			}
		}
		//������̽��ճ������
		else
		{
			MPI_Recv(&m[k][0], N, MPI_FLOAT, k % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		// ��Ԫ����
		int begin = N / size * size + rank < N ? N / size * size + rank : N / size * size + rank - size;
		for (int i = begin; i > k; i -= size) {
			for (int j = k + 1; j < N; j++) {
				m[i][j] = m[i][j] - m[i][k] * m[k][j];
			}
			m[i][k] = 0;
		}
	}
	Tend = MPI_Wtime();
	if (rank == 0)
	{
		//print();
		time2 += (Tend - Tstart) * 1000.0;
	}
	return;
}
int main() {
	MPI_Init(nullptr, nullptr);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//MPI_block����
	for (int i = 0; i < times; i++)
	{
		m_reset();
		MPI_block();
	}
	if (rank == 0) {
		cout << "N=" << N << " MPI_block��" << time1/times << "ms" << endl;
	}
	//MPI_cycle����
	for (int i = 0; i < times; i++)
	{
		m_reset();
		MPI_cycle();
	}
	if (rank == 0) {
		cout << "N=" << N << " MPI_cycle��" << time2 / times << "ms" << endl;
	}
	MPI_Finalize();
	return 0;
}
