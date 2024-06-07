#include <iostream>
#include <mpi.h>
#include <cmath>
#include<sys/time.h>
#include <arm_neon.h>
#include <omp.h>
using namespace std;
const int N = 2048;
float m[N][N];
int NUM_THREADS = 8;//�߳���
const int times = 10;
double timeuse1 = 0, timeuse2 = 0, timeuse3 = 0, timeuse4 = 0;
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
//��ͨMPI
void MPI_cycle() {
    double Tstart, Tend;

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Tstart = MPI_Wtime();
    int tasks;//ÿ�����̵�������
    if (rank < N % size) {
        tasks = N / size + 1;
    }
    else {
        tasks = N / size;
    }
    // 0�Ž��̸�������ַ�
    float* buff = new float[tasks * N];
    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            for (int i = p; i < N; i += size) {
                for (int j = 0; j < N; j++) {
                    buff[i / size * N + j] = m[i][j];
                }
            }
            int ptasks = p < N % size ? N / size + 1 : N / size;  //�����ս��̵���������
            MPI_Send(buff, ptasks * N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
    }
    // ��0�Ž��̸����������
    else {
        MPI_Recv(&m[rank][0], tasks * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //�����յ������ݵ�������Ӧ��λ����
        for (int i = 0; i < tasks; i++) {
            for (int j = 0; j < N; j++) {
                m[rank + i * size][j] = m[rank + i][j];
            }
        }
    }
    // ��ʼ����
    for (int k = 0; k < N; k++) {
        // �������㲢�㲥���
        if (k % size == rank) {
            for (int j = k + 1; j < N; j++) {
                m[k][j] /= m[k][k];
            }
            m[k][k] = 1;
            for (int i = 0; i < size; i++) {
                if (i != rank) {
                    MPI_Send(&m[k][0], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
                }
            }
        }
        // ������̽��ճ������
        else {
            MPI_Recv(&m[k][0], N, MPI_FLOAT, k % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // ������Ԫ
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
        timeuse1 += (Tend - Tstart) * 1000.0;
    }
    return;
}

//MPI��NEON���
void MPI_NEON() {
    double Tstart, Tend;

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Tstart = MPI_Wtime();
    int tasks;//ÿ�����̵�������
    if (rank < N % size) {
        tasks = N / size + 1;
    }
    else {
        tasks = N / size;
    }
    // 0�Ž��̸�������ַ�
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
    for (int k = 0; k < N; k++) {
        // �������㲢�㲥���
        if (k % size == rank) {
            float tmp[4] = { m[k][k], m[k][k], m[k][k], m[k][k] };
            float32x4_t vt = vld1q_f32(tmp);
            vrecpeq_f32(vt);
            int a = k + 1;
            for (int j = k + 1; j + 4 <= N; j += 4, a = j)
            {
                float32x4_t va;
                va = vld1q_f32(m[k] + j);
                va = vmulq_f32(va, vt);
                vst1q_f32(m[k] + j, va);
            }
            for (int j = a; j < N; j++) //ʣ�಻��4�������ִ���
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
            float32x4_t vaik;
            float tmp[4] = { m[i][k], m[i][k], m[i][k], m[i][k] };
            vaik = vld1q_f32(tmp);
            int a = k + 1;
            for (int j = k + 1; j + 4 <= N; j += 4, a = j)
            {
                float32x4_t vakj = vld1q_f32(m[k] + j);
                float32x4_t vaij = vld1q_f32(m[i] + j);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(m[i] + j, vaij);
            }
            for (int j = a; j < N; j++)  //���д�����4����
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
        timeuse2 += (Tend - Tstart) * 1000;
        return;
    }
}

//MPI��OMP���
void MPI_OMP() {
    double Tstart, Tend;

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Tstart = MPI_Wtime();
    int tasks;//ÿ�����̵�������
    if (rank < N % size) {
        tasks = N / size + 1;
    }
    else {
         tasks = N / size;
    }
    // 0�Ž��̸�������ַ�
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
        timeuse3 += (Tend - Tstart) * 1000;
    }
}
// MPI��NEON��OMP���
void MPI_NEON_OMP() {
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
    // 0�Ž��̸�������ַ�
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
                float tmp[4] = { m[k][k], m[k][k], m[k][k], m[k][k] };
                float32x4_t vt = vld1q_f32(tmp);
                vrecpeq_f32(vt);
                int a = k + 1;
                for (int j = k + 1; j + 4 <= N; j += 4, a = j)
                {
                    float32x4_t va;
                    va = vld1q_f32(m[k] + j);
                    va = vmulq_f32(va, vt);
                    vst1q_f32(m[k] + j, va);
                }
                for (int j = a; j < N; j++) //ʣ�಻��4�������ִ���
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
            // ������̽��ճ����еĽ��
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
            float32x4_t vaik;
            float tmp[4] = { m[i][k], m[i][k], m[i][k], m[i][k] };
            vaik = vld1q_f32(tmp);
            int a = k + 1;
            for (int j = k + 1; j + 4 <= N; j += 4, a = j)
            {
                float32x4_t vakj = vld1q_f32(m[k] + j);
                float32x4_t vaij = vld1q_f32(m[i] + j);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(m[i] + j, vaij);
            }
            for (int j = a; j < N; j++)  //���д�����4����
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
        timeuse4 += (Tend - Tstart) * 1000;
    }
}


int main()
{
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
        cout << "N=" << N << " MPI_cycle��" << timeuse1 / times << "ms" << endl;
    }
    //MPI_Neon����
    for (int i = 0; i < times; i++)
    {
        m_reset();
        MPI_NEON();
    }
    if (rank == 0) {
        cout << "N=" << N << " MPI_Neon��" << timeuse2 / times << "ms" << endl;
    }
    //MPI_OMP����
    for (int i = 0; i < times; i++)
    {
        m_reset();
        MPI_OMP();
    }
    if (rank == 0) {
        cout << "N=" << N << " MPI_OMP��" << timeuse3 / times << "ms" << endl;
    }
    //MPI_Neon_OMP����
    for (int i = 0; i < times; i++)
    {
        m_reset();
        MPI_NEON_OMP();
    }
    if (rank == 0) {
        cout << "N=" << N << " MPI_Neon_OMP��" << timeuse4 / times << "ms" << endl;
    }
    MPI_Finalize();
    return 0;
}