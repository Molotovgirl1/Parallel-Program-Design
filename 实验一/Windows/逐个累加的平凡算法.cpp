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
	//定义累加结果
	int sum;
	clock_t begin, end;
	begin = clock();
	for (int cnt = 0;cnt < k;cnt++) {
		sum = 0;
		for (int i = 0;i < n;i++) {
			sum += a[i];
		}
	}
	//cout << sum << endl;
	end = clock();
	float seconds = (end - begin) / float(CLOCKS_PER_SEC);
	cout << "总时间为：" << seconds << endl;
	cout << "平均时间为：" << seconds / k << endl;
	//释放内存
	delete[]a;
	return 0;
}