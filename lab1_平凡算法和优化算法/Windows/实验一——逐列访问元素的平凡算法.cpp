#include<iostream>
using namespace std;
int main() {
	int n;
	int k;
	cout << "��������Թ�ģ��" << endl;
	cin >> k;
	cout << "����������ģ��" << endl;
	cin >> n;
	//����n*n����
	int** b = new int* [n];
	for (int i = 0;i < n;i++) {
		b[i] = new int[n];
	}
	for (int i = 0;i < n;i++) {
		for (int j = 0;j < n;j++) {
			b[i][j] = i + j;
		}
	}
	//��������
	int* a = new int[n];
	//�����Ž��������
	int* sum = new int[n];
	for (int i = 0;i < n;i++) {
		a[i] = i;
	}
	clock_t start, end;
	//ƽ���㷨�����������ڻ�
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
	cout << "��ʱ��Ϊ��" << seconds << endl;
	cout << "ƽ��ʱ��Ϊ:" << seconds / k << endl;
	//�ͷ��ڴ�
	delete[]a;
	for (int i = 0;i < n;i++) {
		delete[]b[i];
	}
	delete[]b;
	delete[]sum;
	return 0;

}