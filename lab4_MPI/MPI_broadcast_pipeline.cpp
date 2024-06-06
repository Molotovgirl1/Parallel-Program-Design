#include<iostream>
#include<Windows.h>
#include<mpi.h>
using namespace std;
const int N = 256;
float m[N][N];
const int times = 1;
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
//MPI:�㲥��ʽ������ѭ�����֣�
void MPI_broadcast() {
    double Tstart, Tend;
    int rank; 
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    //��ʼ��ʱ
    Tstart = MPI_Wtime();
    int tasks; //������������
    //���񻮷�
    if (rank < N % size)
    {
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
            int ptasks = p < N % size ? N / size + 1 : N / size; //�������̵������� 
            MPI_Send(buff, ptasks * N, MPI_FLOAT, p , 0, MPI_COMM_WORLD);
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
        // �������㲢������㲥
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
        // ������̽��ճ����еĽ��
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
        time1 += (Tend - Tstart) * 1000.0;
    }
    return;
}
void MPI_pipeline() {
    double Tstart, Tend;
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    //��ʼ��ʱ
    Tstart = MPI_Wtime();
    int tasks; //���̵�������
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
            int ptasks = p < N % size ? N / size + 1 : N / size; //�������������
            MPI_Send(buff, ptasks * N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
    }
    // ��0�Ž��̽����������
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
    int pre = (rank + (size - 1)) % size; //ǰһ������
    int next = (rank + 1) % size; //��һ������
    for (int k = 0; k < N; k++) {
        // �������㲢�����������һ������
        if (k % size == rank) {
            for (int j = k + 1; j < N; j++) {
                m[k][j] /= m[k][k];
            }
            m[k][k] = 1;
            if (next != rank) {
                MPI_Send(&m[k][0], N, MPI_FLOAT, next, 1, MPI_COMM_WORLD);
            }
        }
        // ������̽��ճ������
        else {
            //����һ�����̽��ս��
            if (pre != rank) {
                MPI_Recv(&m[k][0], N, MPI_FLOAT, pre, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            //�����һ�����������ȷ����������������Ǹ��̣߳����ü�����������ͷ���Ϣ��
            if (next != k % size) {
                MPI_Send(&m[k][0], N, MPI_FLOAT, next, 1, MPI_COMM_WORLD);
            }
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
    //MPI_broadcast����
    for (int i = 0; i < times; i++)
    {
        m_reset();
        MPI_broadcast();
    }
    if (rank == 0) {
        cout << "N=" << N << " MPI_broadcast��" << time1 / times << "ms" << endl;
    }
    //MPI_pipeline����
    for (int i = 0; i < times; i++)
    {
        m_reset();
        MPI_pipeline();
    }
    if (rank == 0) {
        cout << "N=" << N << " MPI_pipeline��" << time2 / times << "ms" << endl;
    }
    MPI_Finalize();
    return 0;
}
