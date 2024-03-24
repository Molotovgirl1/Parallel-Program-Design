#include<iostream>
#include<sys/time.h>
using namespace std;
int main() {
	int n;
	int k;
	cout << "please input the size of test:" << endl;
	cin >> k;
	cout << "please input the size of n*n:" << endl;
	cin >> n;
	int** b = new int* [n];
	for (int i = 0;i < n;i++) {
		b[i] = new int[n];
	}
	for (int i = 0;i < n;i++) {
		for (int j = 0;j < n;j++) {
			b[i][j] = i + j;
		}
	}
	int* a = new int[n];
	int* sum = new int[n];
	for (int i = 0;i < n;i++) {
		a[i] = i;
	}
	struct timeval begin,end;
	//start time
	gettimeofday(&begin,NULL);
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
	}
	/*for (int p = 0;p < n;p++) {
		cout << sum[p] << endl;
		}*/
	//end time
	gettimeofday(&end,NULL);
	double timeuse=(end.tv_sec-begin.tv_sec)*1000+(double)(end.tv_usec-begin.tv_usec)/1000;
	cout<<"total time:"<<timeuse<<"ms"<<endl;
	cout<<"avg time:"<<timeuse/k<<"ms"<<endl;
	delete[]a;
	for (int i = 0;i < n;i++) {
		delete[]b[i];
	}
	delete[]b;
	delete[]sum;
	return 0;

}
