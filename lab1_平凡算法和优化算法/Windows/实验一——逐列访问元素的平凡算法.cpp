#include<iostream>
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
	clock_t start, end;
	//平凡算法矩阵与向量内积
	start = clock();
	for (int i = 0;i < k;i++)
	{
		for (int cnt = 0;cnt < n;cnt++) {
			sum[cnt]=0;
		}
		for (int row = 0;row < n;row++) {
			for (int col = 0;col < n;col++) {
				sum[row] += a[col] * b[col][row];
			}
		}
		/*for (int p = 0;p < n;p++) {
			cout << sum[p] << endl;
		}*/
	}
	end = clock();
	float seconds = (end - start) / float(CLOCKS_PER_SEC);
	cout << "总时间为：" << seconds << endl;
	cout << "平均时间为:" << seconds / k << endl;
	//释放内存
	delete[]a;
	for (int i = 0;i < n;i++) {
		delete[]b[i];
	}
	delete[]b;
	delete[]sum;
	return 0;

}