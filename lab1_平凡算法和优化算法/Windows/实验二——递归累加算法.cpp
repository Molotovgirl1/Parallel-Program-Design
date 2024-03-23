#include<iostream>
#include<Windows.h>
using namespace std;
//递归函数
void recursion(int n, int* a) {
	if (n == 1)
		return;
	for (int i = 0;i < n / 2;i++) {
		a[i] += a[n - 1 - i];
	}
	recursion(n % 2 == 0 ? n / 2 : (n + 1) / 2, a);
}
int main() {
	cout << "请输入测试规模：" << endl;
	int k;
	cin >> k;
	cout << "请输入数组规模：" << endl;
	int n;
	cin >> n;
	//定义数组
	int* a = new int[n];
	for (int i = 0;i < n;i++) {
		a[i] = i;
	}
	long long head, tail, freq; //timers
	//similar to CLOCKS_PER_SEC
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	//start time
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//调用递归函数进行累加
	for (int cnt = 0;cnt < k;cnt++) {
		for (int i = 0;i < n/2;i++) {
			a[i] = i;
		}
		recursion(n, a);
	}
	//cout << a[0] << endl;
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "总时间为：" << (tail - head) * 1000.0 / freq << "ms" << endl;
	cout << "平均时间为:" << (tail - head) * 1000.0 / freq / k << "ms" << endl;
	//释放内存
	delete[]a;
	return 0;
}