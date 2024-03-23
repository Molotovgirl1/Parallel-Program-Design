#include<iostream>
#include<Windows.h>
using namespace std;
int main() {
	cout << "请输入测试规模：" << endl;
	int k;
	cin >> k;
	cout << "请输入数组规模：" << endl;
	int n;
	cin >> n;
	//定义累加的数组
	int* a = new int[n];
	for (int i = 0;i < n;i++) {
		a[i] = i;
	}
	long long head, tail, freq; //timers
	//similar to CLOCKS_PER_SEC
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	//start time
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int cnt = 0;cnt < k;cnt++) {
		//恢复数组值
		for (int i = 0;i < n / 2;i++) {
			a[i] = i;
		}
		for (int m = n;m > 1;m % 2 == 0 ? m = m / 2 : m = (m + 1) / 2) {
			for (int i = 0;i < m / 2;i++) {
				a[i] += a[m - 1 - i];
			}
		}
	}
	//cout << a[0] << endl;
	//end time
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "总时间为：" << (tail - head) * 1000.0 / freq << "ms" << endl;
	cout << "平均时间为:" << (tail - head) * 1000.0 / freq / k << "ms" << endl;
	//释放内存
	delete[]a;
	return 0;
}