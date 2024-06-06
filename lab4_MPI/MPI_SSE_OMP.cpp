#include<iostream>
#include<Windows.h>
#include<mpi.h>
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX��AVX2
#include <omp.h>
using namespace std;
const int N = 256;
float m[N][N];
const int times = 10;
int NUM_THREADS = 6;
double time1 = 0, time2 = 0, time3 = 0, time4 = 0;
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
//��ͨMPI
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
		time1 += (Tend - Tstart) * 1000.0;
	}
	return;
}
//MPI��SSE���
void MPI_SSE() {
	double Tstart, Tend;
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	//��ʼ��ʱ
	Tstart = MPI_Wtime();
	int tasks; //ÿ�����̵�������
	if (rank < N % size) {
		tasks = N / size + 1;
	}
	else {
		tasks = N / size;
	}
	// 0�Ž��̽�������ַ�
	float* buff = new float[tasks * N];
	if (rank == 0) {
		for (int p = 1; p < size; p++) {
			for (int i = p; i < N; i += size) {
				for (int j = 0; j < N; j++) {
					buff[i / size * N + j] = m[i][j];
				}
			}
			int ptasks = p < N % size ? N / size + 1 : N / size; //��������������
			MPI_Send(buff, ptasks * N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
		}
	}
	// ��0�Ž��̸����������
	else {
		MPI_Recv(&m[rank][0], tasks * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		for (int i = 0; i < tasks; i++) {
			for (int j = 0; j < N; j++) {
				m[rank + i * size][j] = m[rank + i][j];
			}
		}
	}
	// ��ʼ����
	for (int k = 0; k < N; k++) {
		// �������㲢�㲥
		if (k % size == rank) {
			float tmp[4] = { m[k][k], m[k][k], m[k][k], m[k][k] };
			__m128 vt = _mm_loadu_ps(tmp);
			int a = k + 1;
			for (int j = k + 1; j + 4 <= N; j += 4, a = j)
			{
				__m128 va;
				va = _mm_loadu_ps(m[k] + j);
				va = _mm_div_ps(va, vt);
				_mm_storeu_ps(m[k] + j, va);
			}
			for (int j = a; j < N; j++)
			{
				m[k][j] = m[k][j] / m[k][k];
			}
			m[k][k] = 1.0;
			for (int p = 0; p < size; p++) {
				if (p != rank) {
					MPI_Send(&m[k][0], N, MPI_FLOAT, p, 1, MPI_COMM_WORLD);
				}
			}
		}
		// ������̽��ճ������
		else {
			MPI_Recv(&m[k][0], N, MPI_FLOAT, k % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		// ������Ԫ����
		int begin = k + 1;
		while (begin % size != rank)  //�ҵ����м����Ŀ�ʼ������
		{
			begin++;
		}
		for (int i = begin; i < N; i += size) {
			__m128 vaik;
			float tmp[4] = { m[i][k], m[i][k], m[i][k], m[i][k] };
			vaik = _mm_loadu_ps(tmp);
			int a = k + 1;
			for (int j = k + 1; j + 4 <= N; j += 4, a = j)
			{

				__m128 vakj = _mm_loadu_ps(m[k] + j);
				__m128 vaij = _mm_loadu_ps(m[i] + j);
				__m128 vx = _mm_mul_ps(vakj, vaik);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_storeu_ps(m[i] + j, vaij);
			}
			for (int j = a; j < N; j++)
			{
				m[i][j] = m[i][j] - m[k][j] * m[i][k];
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
//MPI��OMP���
void MPI_OMP() {
	double Tstart, Tend;
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	//��ʼ��ʱ
	Tstart = MPI_Wtime();
	int tasks; //ÿ�����̵�������
	if (rank < N % size) {
		tasks = N / size + 1;
	}
	else {
		tasks = N / size;
	}
	// 0�Ž��̽�������ַ�
	float* buff = new float[tasks * N];
	if (rank == 0) {
		for (int p = 1; p < size; p++) {
			for (int i = p; i < N; i += size) {
				for (int j = 0; j < N; j++) {
					buff[i / size * N + j] = m[i][j];
				}
			}
			int ptasks = p < N % size ? N / size + 1 : N / size;
			MPI_Send(buff, ptasks * N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
		}
	}
	// ��0�Ž��̸����������
	else {
		MPI_Recv(&m[rank][0], tasks * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		for (int i = 0; i < tasks; i++) {
			for (int j = 0; j < N; j++) {
				m[rank + i * size][j] = m[rank + i][j];
			}
		}
	}
	// ��ʼ����
	int i, j, k;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k) shared(m, N, size, rank)
	for (k = 0; k < N; k++) {
		// �������㲢�㲥���
#pragma omp single
		{
			if (k % size == rank) {
				for (j = k + 1; j < N; j++) {
					m[k][j] /= m[k][k];
				}
				m[k][k] = 1;
				for (int p = 0; p < size; p++) {
					if (p != rank) {
						MPI_Send(&m[k][0], N, MPI_FLOAT, p, 1, MPI_COMM_WORLD);
					}
				}
			}
			// ������̽��ճ������
			else {
				MPI_Recv(&m[k][0], N, MPI_FLOAT, k % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
		// ������Ԫ����
		int begin = k + 1;
		while (begin % size != rank)  //�ҵ����м����Ŀ�ʼ������
		{
			begin++;
		}
#pragma omp for schedule(simd : guided)
		for (int i = begin; i < N; i += size) {
			for (j = k + 1; j < N; j++) {
				m[i][j] = m[i][j] - m[i][k] * m[k][j];
			}
			m[i][k] = 0;
		}
	}
	Tend = MPI_Wtime();
	if (rank == 0)
	{
		//print();
		time3 += (Tend - Tstart) * 1000.0;
	}
}
//MPI��SSE��OMP���
void MPI_SSE_OMP() {
	double Tstart, Tend;
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	//��ʼ��ʱ
	Tstart = MPI_Wtime();
	int tasks;//ÿ�����̵�������
	if (rank < N % size) {
		tasks = N / size + 1;
	}
	else {
		tasks = N / size;
	}
	// 0�Ž��̽�������ַ�
	float* buff = new float[tasks * N];
	if (rank == 0) {
		for (int p = 1; p < size; p++) {
			for (int i = p; i < N; i += size) {
				for (int j = 0; j < N; j++) {
					buff[i / size * N + j] = m[i][j];
				}
			}
			int ptasks = p < N % size ? N / size + 1 : N / size;
			MPI_Send(buff, ptasks * N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
		}
	}
	// ��0�Ž��̸����������
	else {
		MPI_Recv(&m[rank][0], tasks * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		for (int i = 0; i < tasks; i++) {
			for (int j = 0; j < N; j++) {
				m[rank + i * size][j] = m[rank + i][j];
			}
		}
	}
	// ��ʼ����
	int i, j, k;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k) shared(m, N, size, rank)
	for (k = 0; k < N; k++) {
		//�������㲢�㲥���
#pragma omp single
		{
			if (k % size == rank) {
				float tmp[4] = { m[k][k], m[k][k], m[k][k], m[k][k] };
				__m128 vt = _mm_loadu_ps(tmp);
				int a = k + 1;
				for (int j = k + 1; j + 4 <= N; j += 4, a = j)
				{
					__m128 va;
					va = _mm_loadu_ps(m[k] + j);
					va = _mm_div_ps(va, vt);
					_mm_storeu_ps(m[k] + j, va);
				}
				for (int j = a; j < N; j++)
				{
					m[k][j] = m[k][j] / m[k][k];
				}
				m[k][k] = 1.0;
				for (int p = 0; p < size; p++) {
					if (p != rank) {
						MPI_Send(&m[k][0], N, MPI_FLOAT, p, 1, MPI_COMM_WORLD);
					}
				}
			}
			// ������̽��ճ������
			else {
				MPI_Recv(&m[k][0], N, MPI_FLOAT, k % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
		// ������Ԫ����
		int begin = k + 1;
		while (begin % size != rank)  //�ҵ����м����Ŀ�ʼ������
		{
			begin++;
		}
#pragma omp for schedule(simd : guided)
		for (int i = begin; i < N; i += size) {
			__m128 vaik;
			float tmp[4] = { m[i][k], m[i][k], m[i][k], m[i][k] };
			vaik = _mm_loadu_ps(tmp);
			int a = k + 1;
			for (int j = k + 1; j + 4 <= N; j += 4, a = j)
			{

				__m128 vakj = _mm_loadu_ps(m[k] + j);
				__m128 vaij = _mm_loadu_ps(m[i] + j);
				__m128 vx = _mm_mul_ps(vakj, vaik);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_storeu_ps(m[i] + j, vaij);
			}
			for (int j = a; j < N; j++)
			{
				m[i][j] = m[i][j] - m[k][j] * m[i][k];
			}
			m[i][k] = 0;
		}
	}
	Tend = MPI_Wtime();
	if (rank == 0)
	{
		//print();
		time4 += (Tend - Tstart) * 1000.0;
	}
}

int main() {
	MPI_Init(nullptr, nullptr);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//MPI_cycle����
	for (int i = 0; i < times; i++)
	{
		m_reset();
		MPI_cycle();
	}
	if (rank == 0) {
		cout << "N=" << N << " MPI_cycle��" << time1 / times << "ms" << endl;
	}
	//MPI_SSE����
	for (int i = 0; i < times; i++)
	{
		m_reset();
		MPI_SSE();
	}
	if (rank == 0) {
		cout << "N=" << N << " MPI_SSE��" << time2 / times << "ms" << endl;
	}
	//MPI_OMP����
	for (int i = 0; i < times; i++)
	{
		m_reset();
		MPI_OMP();
	}
	if (rank == 0) {
		cout << "N=" << N << " MPI_OMP��" << time3 / times << "ms" << endl;
	}
	//MPI_SSE_OMP����
	for (int i = 0; i < times; i++)
	{
		m_reset();
		MPI_SSE_OMP();
	}
	if (rank == 0) {
		cout << "N=" << N << " MPI_SSE_OMP��" << time4 / times << "ms" << endl;
	}
	MPI_Finalize();
	return 0;
}