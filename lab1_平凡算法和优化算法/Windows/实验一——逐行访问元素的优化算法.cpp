#include<iostream>
#include<Windows.h>
using namespace std;
int main() {
	int n;
	int k;
	cout << "请输入测试规模：" << endl;
	cin >> k;
	cout << "请输入矩阵规模：" << endl;
	cin >> n;
	//定义n*n矩阵
	int** b = new int* [n];
	for (int i = 0;i < n;i++) {
		b[i] = new int[n];
	}
	for (int i = 0;i < n;i++) {
		for (int j = 0;j < n;j++) {
			b[i][j] = i + j;
		}
	}
	//定义向量
	int* a = new int[n];
	//定义存放结果的向量
	int* sum = new int[n];
	for (int i = 0;i < n;i++) {
		a[i] = i;
	}
	long long head, tail, freq; //timers
	//similar to CLOCKS_PER_SEC
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq); 
	//start time
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//优化算法矩阵与向量内积
	for (int i = 0;i < k;i++)
	{
		for (int cnt = 0;cnt < n;cnt++) {
			sum[cnt] = 0;
		}
		for (int row = 0;row < n;row++) {
			for (int col = 0;col < n;col++) {
				sum[col] += a[row] * b[row][col];
			}
		}
		/*for (int p = 0;p < n;p++) {
			cout << sum[p] << endl;
		}*/
	}
	//end time
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "总时间为：" << (tail - head) * 1000.0 / freq <<"ms"<< endl;
	cout << "平均时间为:" << (tail-head )*1000.0/freq/ k <<"ms"<< endl;
	//释放内存
	delete[]a;
	for (int i = 0;i < n;i++) {
		delete[]b[i];
	}
	delete[]b;
	delete[]sum;
	return 0;

}