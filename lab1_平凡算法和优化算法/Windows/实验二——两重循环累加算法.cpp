#include<iostream>
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
	clock_t begin, end;
	begin = clock();
	for (int cnt = 0;cnt < k;cnt++) {
		//�ָ�����ֵ
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
	cout << "��ʱ��Ϊ��" << seconds << endl;
	cout << "ƽ��ʱ��Ϊ��" << seconds / k << endl;
	//�ͷ��ڴ�
	delete[]a;
	return 0;
}