#include<iostream>
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
	clock_t begin, end;
	begin = clock();
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
	end = clock();
	float seconds = (end - begin) / float(CLOCKS_PER_SEC);
	cout << "总时间为：" << seconds << endl;
	cout << "平均时间为：" << seconds / k << endl;
	//释放内存
	delete[]a;
	return 0;
}