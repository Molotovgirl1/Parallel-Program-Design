#include<iostream>
#include<sys/time.h>
using namespace std;
int main() {
	cout << "please input the size of test:" << endl;
	int k;
	cin >> k;
	cout << "please input the size of array:" << endl;
	int n;
	cin >> n;
	int* a = new int[n];
	for (int i = 0;i < n;i++) {
		a[i] = i;
	}
	int sum;
	struct timeval begin,end;
	//start time
	gettimeofday(&begin,NULL);
	for (int cnt = 0;cnt < k;cnt++) {
		sum = 0;
		for (int i = 0;i < n;i++) {
			sum += a[i];
		}
	}
	//cout << sum << endl;
	//end time
	gettimeofday(&end,NULL);
	double timeuse=(end.tv_sec-begin.tv_sec)*1000+(double)(end.tv_usec-begin.tv_usec)/1000;
	cout<<"total time:"<<timeuse<<"ms"<<endl;
	cout<<"avg time:"<<timeuse/k<<"ms"<<endl;
	delete[]a;
	return 0;
}
