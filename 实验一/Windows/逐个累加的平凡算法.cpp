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
	//�����ۼӽ��
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
	cout << "��ʱ��Ϊ��" << seconds << endl;
	cout << "ƽ��ʱ��Ϊ��" << seconds / k << endl;
	//�ͷ��ڴ�
	delete[]a;
	return 0;
}