#include <iostream>
#include <CL/sycl.hpp>
#include <chrono>

#define random_float() (rand() / double(RAND_MAX))
const int N=4096;

using namespace std;
using namespace sycl;

void print(int N, float** m)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            cout << m[i][j] << " ";
        printf("\n");
    }
}

// 生成测试用例
void m_reset(int N, float** m) {
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
        {
            m[i][j] = 0;
        }
        m[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
        {
            m[i][j] = random_float();
        }
    }

    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1; i < N; i++)
        {
            for (int j = 0; j < N; j++)
                m[i][j] += m[k][j];
        }
    }
}

// CPU算法
void cpu_elimination(int N, float** m) {
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

// GPU算法
void gpu_elimination(int N, float** m, queue& q) {
    auto M = range<1>(N);
    auto M_minus_k = range<1>(N - 1);

    // Create buffers for matrix
    buffer<float, 2> buf_m(*m, range<2>(N, N));

    q.submit([&](handler& h) {
        auto acc_m = buf_m.get_access<access::mode::read_write>(h);

        h.parallel_for(M, [=](id<1> k) {
            for (int j = k + 1; j < N; j++)
                acc_m[k][j] /= acc_m[k][k];
            acc_m[k][k] = 1.0f;

            for (int i = k + 1; i < N; i++) {
                for (int j = k + 1; j < N; j++) {
                    acc_m[i][j] -= acc_m[i][k] * acc_m[k][j];
                }
                acc_m[i][k] = 0.0f;
            }
        });
    });

    q.wait();
}

int main() {
    
        float** m_cpu = new float*[N];
        float** m_gpu = new float*[N];

        for (int i = 0; i < N; i++) {
            m_cpu[i] = new float[N];
            m_gpu[i] = new float[N];
        }

        m_reset(N, m_cpu);  //初始化矩阵
        memcpy(m_gpu, m_cpu, N * sizeof(float*));  //确保

        // CPU
        auto start_cpu = chrono::high_resolution_clock::now();
        cpu_elimination(N, m_cpu);
        auto end_cpu = chrono::high_resolution_clock::now();
        double time_cpu = chrono::duration_cast<chrono::milliseconds>(end_cpu - start_cpu).count();

        //GPU
        queue q(gpu_selector_v);
        auto start_gpu = chrono::high_resolution_clock::now();
        gpu_elimination(N, m_gpu, q);
        auto end_gpu = chrono::high_resolution_clock::now();
        double time_gpu = chrono::duration_cast<chrono::milliseconds>(end_gpu - start_gpu).count();

        cout << "N=" << N << "  CPU time: " << time_cpu << " ms  GPU time: " << time_gpu << " ms" ;
      
        // 验证计算结果(optional)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if( fabs(m_cpu[i] - m_gpu[i]) > 1e-3) {
                    cout << "Results differ!";
                    break;
                }
            }
        }

    return 0;
}