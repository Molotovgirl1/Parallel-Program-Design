#include<iostream>
#include<fstream>
#include<sstream>
#include<bitset>
#include<cmath>
#include<sys/time.h>
#include<semaphore.h>
#include<cstring>
#include <stdio.h>
using namespace std;

const int colNum = 130;
const int R_lineNum = colNum;
const int actual_colNum = ceil(colNum * 1.0 / 32);

int eNum;
int* First;
unsigned int** R;
unsigned int** E;

void Init_Zero(unsigned int** m, int line, int col)
{
	for (int i = 0; i < line; i++)
	{
		for (int j = 0; j < col; j++)
		{
			m[i][j] = 0;
		}
	}
}

/*初始化E被消元行矩阵，从txt中获取被消元行矩阵*/
void Init_E(unsigned int** E, int* First)
{
	unsigned int StringToNum;
	ifstream infile("E.txt");
	if (!infile.is_open())
		cerr << " wrong when open E.txt! " << endl;

	char fin[20000] = { 0 };
	int index = 0;
	while (infile.getline(fin, sizeof(fin))) {
		std::stringstream line(fin);
		int IsFirst = 0;         //用于判断是不是首项
		while (line >> StringToNum) {     //是首项则存入First数组
			if (IsFirst == 0) {
				First[index] = StringToNum;
				IsFirst = 1;
			}
			int offset = StringToNum % 32;
			int post = StringToNum / 32;
			int temp = 1 << offset;
			E[index][actual_colNum - 1 - post] += temp;
		}
		index++;
	}
	infile.close();
}

/*初始化R消元子矩阵，从txt中获取消元子矩阵*/
void Init_R(unsigned int** R)
{

	unsigned int StringToNum;
	ifstream infile("R.txt");
	if (!infile.is_open())
		cerr << " wrong when open R.txt! " << endl;

	char fin[20000] = { 0 };
	int index = 0;
	while (infile.getline(fin, sizeof(fin))) {
		std::stringstream line(fin);
		int IsFirst = 0;         //用于判断是不是首项
		while (line >> StringToNum) {
			if (IsFirst == 0) {
				index = StringToNum;   //对于消元子来说首项是谁就放到哪一行里面，有很多空行
				IsFirst = 1;
			}
			int offset = StringToNum % 32;   //+1的原因是txt中给的列数是从0开始计数而不是1
			int post = StringToNum / 32;
			int temp = 1 << offset;
			R[index][actual_colNum - 1 - post] += temp;
		}
	}
	infile.close();
}

/*判断R消元子矩阵的某一行是否为空*/
bool Is_R_Null(unsigned int* R, int index)
{
	for (int i = 0; i < actual_colNum; i++)
	{
		if (R[index+ actual_colNum+i] != 0)
			return false;
	}
	return true;
}


/*当R消元子矩阵中没有E首项对应的行时，E的那一行完成消元，同时设置在R对应的行成为消元子*/
void Set_EtoR(unsigned int* R, unsigned int* E, int Eindex, int Rindex)
{
	for (int i = 0; i < actual_colNum; i++)
	{
		R[Rindex+*actual_colNum+i] = E[Eindex* actual_colNum+i];
	}
}

//设备端 判断消元子是否为空
__device__ bool Is_gR_Null(unsigned int* R, int index, int col)
{
	for (int i = 0; i < col; i++)
	{
		if (R[col * index + i] != 0)
			return false;
	}
	return true;
}
//设备端更改首项
__device__ void  Reset_gEfirst(unsigned int* E, int index, int* First, int col)
{
	int i = 0;
	while (E[index * col + i] == 0 && i < col)
	{
		i++;
	}
	if (i == actual_colNum)
	{
		First[index] = -1;
		return;
	}
	unsigned int temp = E[index * col + i];
	int j = 0;
	while (temp != 0)
	{
		temp = temp >> 1;
		j++;
	}
	First[index] = col * 32 - (i + 1) * 32 + j - 1;
}

__global__ void work(int g_actual_colNum, int g_eNum, int g_R_lineNum, int ∗ g_R, int ∗g_E, int* First)
{
	int g_index = blockIdx.x * blockDim.x + threadIdx.x;
	int gridStride = gridDim.x * blockDim.x;

	for (int i = g_R_lineNum - 1; i - 5 >= -1; i -= 5)
	{
		for (int j = g_index; j < g_eNum; j += gridStride)
		{
			while (First[j] <= i && First[j] >= i - 4)
			{
				int index = Fisrt[j];
				if (!Is_gR_Null(g_R,index, g_actual_colNum))  //消元子非空
				{
					for (int k = 0; k < g_actual_colNum; k++)  //异或消元
					{
						  g_E[j ∗g_actual_colNum + k] = g_E[j ∗g_actual_colNum + k] ^g_R[index ∗g_actual_colNum + k];
					 }
					Reset_gEfirst(g_E, index, First, g_actual_colNum);  //更新首项
				 }
				else
					break;

			}

		}

	}
	for (int i = g_R_lineNum % 5-1; i >= 0; i--)
	{
		for (int j = g_index; j < g_eNum; j += gridStride)
		{
			while (First[j]==i)
			{
				int index = Fisrt[j];
				if (!Is_gR_Null(g_R, index, g_actual_colNum))  //消元子非空
				{
					for (int k = 0; k < g_actual_colNum; k++)  //异或消元
					{
						g_E[j ∗g_actual_colNum + k] = g_E[j ∗g_actual_colNum + k] ^ g_R[index ∗g_actual_colNum + k];
					}
					Reset_gEfirst(g_E, index, First, g_actual_colNum);  //更新首项
				}
				else
					break;
			}
		}
	}
}

int main()
{

	/*eNum中存放被消元行的行数*/
	ifstream infile("E.txt");
	if (!infile.is_open())
		cerr << " wrong! " << endl;
	char fin[20000] = { 0 };
	//int eNum = 0;
	while (infile.getline(fin, sizeof(fin)))
	{
		eNum++;
	}
	infile.close();


	/*由于存E矩阵的首项*/
	int* First = new int[eNum];

	/*E矩阵是被消元行矩阵*/
	unsigned int** E = new unsigned int* [eNum];
	for (int i = 0; i < eNum; ++i)
		E[i] = new unsigned int[actual_colNum];
	Init_Zero(E, eNum, actual_colNum);
	Init_E(E, First);

	/*R矩阵是消元子矩阵*/
	unsigned int** R = new unsigned int* [R_lineNum];
	for (int i = 0; i < R_lineNum; ++i)
		R[i] = new unsigned int[actual_colNum];
	Init_Zero(R, R_lineNum, actual_colNum);
	Init_R(R);

	unsigned int* E_flat = new unsigned int[eNum * actual_colNum];

	// 将二维数组 E 展开为一维数组 E_flat
	for (int i = 0; i < eNum; i++) {
		for (int j = 0; j < actual_colNum; j++) {
			E_flat[i * actual_colNum + j] = E[i][j];
		}
	}

	unsigned int* R_flat = new unsigned int[R_lineNum * actual_colNum];
	// 将二维数组 R 展开为一维数组 R_flat
	for (int i = 0; i < R_lineNum; i++) {
		for (int j = 0; j < actual_colNum; j++) {
			R_flat[i * actual_colNum + j] = R[i][j];
		}
	}

	/*---------------------------以上完成cpu端数据的导入与初始化----------------------------*/


	cudaError_t ret;
	unsigned int ∗ g_R, ∗g_E,*g_First;

	ret = cudaMalloc(&g_E, eNum ∗actual_colNum ∗ sizeof(int));
    ret = cudaMalloc(&g_R, R_lineNum ∗actual_colNum ∗ sizeof(int));
	ret = cudaMalloc(&g_First,eNum ∗ sizeof(int));

	 if(ret != cudaSuccess) {
		printf(" cudaMalloc gpudata failed !\n");
	 }
    size_t threads_per_block = 256;
	size_t number_of_blocks = 32;

	 cudaEvent_t start, stop; 
	 float etime = 0.0;
	 cudaEventCreate(&start);
	 cudaEventCreate(&stop);
	 cudaEventRecord(start, 0); 

	 bool sign;
	 do
	 {
		 ret = cudaMemcpy(g_E, E_flat, eNum ∗actual_colNum ∗ sizeof(int), cudaMemcpyHostToDevice);
		 ret = cudaMemcpy(g_R, R_flat, R_lineNum ∗actual_colNum ∗ sizeof(int), cudaMemcpyHostToDevice);
		 ret = cudaMemcpy(g_First, First, eNum ∗ sizeof(int), cudaMemcpyHostToDevice);
		 if(ret != cudaSuccess) {
			printf(" cudaMemcpyHostToDevice failed !\n");
		 }

		 //消元，不设置消元子
		 work << < 1024, 10 >> > (actual_colNum, eNum, R_lineNum, g_R, g_E,g_First);
		 cudaDeviceSynchronize();

		 ret = cudaMemcpy(E_flat, g_E, eNum ∗actual_colNum ∗ sizeof(int),cudaMemcpyDeviceToHost);
		 ret = cudaMemcpy(R_flat, g_R, R_lineNum ∗actual_colNum ∗ sizeof(int), cudaMemcpyDeviceToHost);
		 ret = cudaMemcpy(First, g_First, eNum ∗ sizeof(int), cudaMemcpyDeviceToHost);
		  if(ret != cudaSuccess) {
			  printf(" cudaMemcpyDeviceToHost failed !\n");
		  }

		  sign = false;
		  for (int i = 0; i < eNum; i++)
		  {
			  if(First[i] == -1)
				  continue;
			  if (Is_R_Null(R, First[i]))
			  {
				  Set_EtoR(R, E, i, First[i]);
				  First[i] = -1;
				  sign = true;
			  }
		  }
	 } while (sign == true);

	 cudaEventRecord(stop, 0);
	 cudaEventSynchronize(stop); 
	 cudaEventElapsedTime(&etime, start, stop);
	 printf("GPU :%f ms\n", etime);
}