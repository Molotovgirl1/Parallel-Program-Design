#include<iostream>
#include<fstream>
#include<sys/time.h>
#include<arm_neon.h>
#include<sstream>
#include<cmath>
#include<bitset>
using namespace std;
const int times = 100;
const int ColNum = 130;  //矩阵的列数
const int R_RowNum = ColNum; 
const int R_ColNum = ceil(ColNum * 1.0 / 32);  
const int E_ColNum = ceil(ColNum * 1.0 / 32); 
string datapath = "/home/s2110508/SIMD/data/Grobner/1_130_22_8/";
//矩阵初始化
void Init_Zero(unsigned int** m, int row, int col) {
    for (int i = 0;i < row;i++) {
        for (int j = 0;j < col;j++) {
            m[i][j] = 0;
        }
    }
}
//从txt文件中读取数据初始化被消元行矩阵E
void Init_E(unsigned int** E, int* First)
{
    ifstream file(datapath+"2.txt"); 
    if (!file.is_open()) {
        cerr << "Failed to open the E file." << endl;
        return ;
    }
    string line;
    int index = 0;
    while (getline(file, line)) {
        istringstream iss(line);
        unsigned int number;
        bool isFirst = false;
        while (iss >> number) {
            if (isFirst == false) { //判断是不是首项
                First[index] = number;
                isFirst = true;
            }
            //将数字存入矩阵中
            int offset = number % 32;
            int post = number / 32;
            int temp = 1 << offset;
            E[index][E_ColNum - 1 - post] += temp;
        }
        index++;
    }
    file.close();
}
//从txt文件中读取数据初始化消元行矩阵R
void Init_R(unsigned int** R)
{
    ifstream file(datapath+"1.txt");
    if (!file.is_open()) {
        cerr << "Failed to open the R file." << endl;
        return;
    }
    string line;
    int index = 0;
    while (getline(file, line)) {
        istringstream iss(line);
        int number;
        bool isFirst = false;
        while (iss >> number) {
            if (isFirst == false) { //判断是不是首项
                index = number; //首项位置决定存放的行位置
                isFirst = true;
            }
            //将数字存入对应行中
            int offset = number % 32;
            int post = number / 32;
            int temp = 1 << offset;
            R[index][E_ColNum - 1 - post] += temp;
        }
    }
    file.close();
}
//判断R消元子矩阵的某一行是否为空
bool Is_R_Null(unsigned int** R, int index)
{
    for (int i = 0; i < R_ColNum; i++)
    {
        if (R[index][i] != 0)
            return false;
    }
    return true;
}
//当R消元子矩阵中没有E首项对应的行时，将E对应行设置在R对应的行成为消元子
void Set_EtoR(unsigned int** R, unsigned int** E, int Eindex, int Rindex)
{
    for (int i = 0; i < R_ColNum; i++)
    {
        R[Rindex][i] = E[Eindex][i];
    }
}
/*重置E被消元行矩阵某一行的首项值*/
void Reset_Efirst(unsigned int** E, int index, int* First)
{
    int i;
    for (i = 0;i < E_ColNum;i++) {
        if (E[index][i] != 0) {
            break;
        }
    }
    if (i == E_ColNum) //该行全为0，没有首项
    {
        First[index] = -1;
        return;
    }
    unsigned int temp = E[index][i];
    int j = 0;
    while (temp != 0)
    {
        temp = temp >> 1;
        j++;
    }
    First[index] = E_ColNum * 32 - (i + 1) * 32 + j - 1; //算出对应首项的值存入First中
   
}
//被消元行与首项对应位置的消元子行进行运算——异或操作
void ExorR(unsigned int** R, unsigned int** E, int Eindex, int Rindex)
{
    for (int i = 0; i < E_ColNum; i++)
    {
        E[Eindex][i] = E[Eindex][i] ^ R[Rindex][i];
    }
}
//串行算法
void serial(int E_RowNum, unsigned int** R, unsigned int** E, int* First)
{
    for (int i = 0; i < E_RowNum; i++)
    {
        while (First[i] != -1)  
        {
            if (Is_R_Null(R, First[i]))  
            {
                Set_EtoR(R, E, i, First[i]);
                break;
            }
            else   
            {
                ExorR(R, E, i, First[i]);
                Reset_Efirst(E, i, First);
            }
        }
    }
}
//对异或运算进行Neon向量化
void Neon_ExorR(unsigned int** R, unsigned int** E, int Eindex, int Rindex) {
    int num = 0;
    for (int i = 0; i + 4 <= E_ColNum; i += 4, num = i) {
        uint32x4_t tmp_E = vld1q_u32(E[Eindex] + i);
        uint32x4_t tmp_R = vld1q_u32(R[Rindex] + i);
        tmp_E = veorq_u32(tmp_E, tmp_R);
        vst1q_u32(E[Eindex] + i, tmp_E);
    }
    for (int i = num;i < E_ColNum;i++) {
        E[Eindex][i] ^= R[Rindex][i];
    }
}
//Neon算法
void Neon(int E_RowNum, unsigned int** R, unsigned int** E, int* First) {
    for (int i = 0; i < E_RowNum; i++)
    {
        while (First[i] != -1)
        {
            if (Is_R_Null(R, First[i]))
            {
                Set_EtoR(R, E, i, First[i]);
                break;
            }
            else
            {
                Neon_ExorR(R, E, i, First[i]);
                Reset_Efirst(E, i, First);
            }
        }
    }
}
//将消元结果存入文件
void save_result(unsigned int** E, int* First, int E_RowNum )
{
    ofstream outputFile(datapath+"res.txt");
    if (!outputFile) {
        cerr << "Failed to open the Result File！" << endl;
        return ;
    }
    for (int i = 0; i < E_RowNum; i++)
    {
        if (First[i] == -1) //对应行全为0
        {
            outputFile << endl;
            //cout << endl;
            continue;
        }
        for (int j = 0; j < E_ColNum; j++)
        {
            if (E[i][j] == 0) {
                continue;
            }  
            int number = 0;
            bitset<32> Bit = E[i][j];
            for (int k = 31; k >= 0; k--) 
            {
                if (Bit.test(k)) //判断Bit的第 k 位是否为1
                {
                    number= 32 * (E_ColNum - j - 1) + k ;
                    //cout << number << " ";
                    outputFile << number << " ";
                }
            }
        }
        //cout << endl;
        outputFile<<endl;
    }
    outputFile.close();
}
int main() {
    struct timeval begin, end;
    double timeuse1=0, timeuse2=0;
    //计算被消元行矩阵的行数
    ifstream file(datapath+"2.txt"); 
    if (!file.is_open()) {
        cerr << "Failed to open the file." << endl;
        return 1;
    }
    int E_RowNum = 0;
    string line;
    while (getline(file, line)) {
        E_RowNum++;
    }
    file.close();
    //定义被消元矩阵E和首项First、消元行矩阵R
    int* First = new int[E_RowNum];
    unsigned int** E = new unsigned int* [E_RowNum];
    for (int i = 0;i < E_RowNum;i++) {
        E[i] = new unsigned int[E_ColNum];
    }
    unsigned int** R = new unsigned int* [R_RowNum];
    for (int i = 0;i < R_RowNum;i++) {
        R[i] = new unsigned int[R_ColNum];
    }
    //串行算法计时
    for (int i = 0;i < times;i++) {
        Init_Zero(E, E_RowNum, E_ColNum);
        Init_E(E, First);
        Init_Zero(R, R_RowNum, R_ColNum);
        Init_R(R);
        gettimeofday(&begin, NULL);
        serial(E_RowNum, R, E, First);
        gettimeofday(&end, NULL);
        timeuse1 += (end.tv_sec - begin.tv_sec) * 1000 + (double)(end.tv_usec - begin.tv_usec) / 1000.0;        
    }
    cout << "col=" << ColNum << "  serial:" << timeuse1 / times << "ms"<<endl;
    //Neon算法计时
    for (int i = 0;i < times;i++) {
        Init_Zero(E, E_RowNum, E_ColNum);
        Init_E(E, First);
        Init_Zero(R, R_RowNum, R_ColNum);
        Init_R(R);
        gettimeofday(&begin, NULL);
        serial(E_RowNum, R, E, First);
        gettimeofday(&end, NULL);
        timeuse2 += (end.tv_sec - begin.tv_sec) * 1000 + (double)(end.tv_usec - begin.tv_usec) / 1000.0;
    }
    cout << "col=" << ColNum << "  Neon:" << timeuse2 / times << "ms"<<endl;
    save_result(E, First, E_RowNum);
	return 0;
}