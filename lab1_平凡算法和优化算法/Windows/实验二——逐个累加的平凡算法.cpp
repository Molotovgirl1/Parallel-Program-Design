#include<iostream>
#include<Windows.h>
using namespace std;
int main() {
	cout << "��������Թ�ģ��" << endl;
	int k;
	cin >> k;
	cout << "�����������ģ��" << endl;
	int n;
	cin >> n;
	//�����ۼӵ�����
	int* a = new int[n];
	for (int i = 0;i < n;i++) {
		a[i] = i;
	}
	//�����ۼӽ��
	int sum;
	long long head, tail, freq; //timers
	//similar to CLOCKS_PER_SEC
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	//start time
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int cnt = 0;cnt < k;cnt++) {
		sum = 0;
		for (int i = 0;i < n;i++) {
			sum += a[i];
		}
	}
	//cout << sum << endl;
	//end time
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "��ʱ��Ϊ��" << (tail - head) * 1000.0 / freq << "ms" << endl;
	cout << "ƽ��ʱ��Ϊ:" << (tail - head) * 1000.0 / freq / k << "ms" << endl;
	//�ͷ��ڴ�
	delete[]a;
	return 0;
}