#include<iostream>
#include<Windows.h>
using namespace std;
//�ݹ麯��
void recursion(int n, int* a) {
	if (n == 1)
		return;
	for (int i = 0;i < n / 2;i++) {
		a[i] += a[n - 1 - i];
	}
	recursion(n % 2 == 0 ? n / 2 : (n + 1) / 2, a);
}
int main() {
	cout << "��������Թ�ģ��" << endl;
	int k;
	cin >> k;
	cout << "�����������ģ��" << endl;
	int n;
	cin >> n;
	//��������
	int* a = new int[n];
	for (int i = 0;i < n;i++) {
		a[i] = i;
	}
	long long head, tail, freq; //timers
	//similar to CLOCKS_PER_SEC
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	//start time
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//���õݹ麯�������ۼ�
	for (int cnt = 0;cnt < k;cnt++) {
		for (int i = 0;i < n/2;i++) {
			a[i] = i;
		}
		recursion(n, a);
	}
	//cout << a[0] << endl;
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "��ʱ��Ϊ��" << (tail - head) * 1000.0 / freq << "ms" << endl;
	cout << "ƽ��ʱ��Ϊ:" << (tail - head) * 1000.0 / freq / k << "ms" << endl;
	//�ͷ��ڴ�
	delete[]a;
	return 0;
}