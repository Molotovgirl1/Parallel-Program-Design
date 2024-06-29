#include<iostream>
#include<fstream>
#include<sstream>
#include<cmath>
#include<bitset>
#include<Windows.h>
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX��AVX2
using namespace std;
const int times = 100;
const int ColNum = 2362;  //���������
const int R_RowNum = ColNum;
const int R_ColNum = ceil(ColNum * 1.0 / 32);
const int E_ColNum = ceil(ColNum * 1.0 / 32);
string datapath = "C:/Users/ybx/Desktop/fsdownload/Grobner/5_2362_1226_453/";
//�����ʼ��
void Init_Zero(unsigned int** m, int row, int col) {
    for (int i = 0;i < row;i++) {
        for (int j = 0;j < col;j++) {
            m[i][j] = 0;
        }
    }
}
//��txt�ļ��ж�ȡ���ݳ�ʼ������Ԫ�о���E
void Init_E(unsigned int** E, int* First)
{
    ifstream file(datapath + "2.txt");
    if (!file.is_open()) {
        cerr << "Failed to open the E file." << endl;
        return;
    }
    string line;
    int index = 0;
    while (getline(file, line)) {
        istringstream iss(line);
        unsigned int number;
        bool isFirst = false;
        while (iss >> number) {
            if (isFirst == false) { //�ж��ǲ�������
                First[index] = number;
                isFirst = true;
            }
            //�����ִ��������
            int offset = number % 32;
            int post = number / 32;
            int temp = 1 << offset;
            E[index][E_ColNum - 1 - post] += temp;
        }
        index++;
    }
    file.close();
}
//��txt�ļ��ж�ȡ���ݳ�ʼ����Ԫ�о���R
void Init_R(unsigned int** R)
{
    ifstream file(datapath + "1.txt");
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
            if (isFirst == false) { //�ж��ǲ�������
                index = number; //����λ�þ�����ŵ���λ��
                isFirst = true;
            }
            //�����ִ����Ӧ����
            int offset = number % 32;
            int post = number / 32;
            int temp = 1 << offset;
            R[index][R_ColNum - 1 - post] += temp;
        }
    }
    file.close();
}
//�ж�R��Ԫ�Ӿ����ĳһ���Ƿ�Ϊ��
bool Is_R_Null(unsigned int** R, int index)
{
    for (int i = 0; i < R_ColNum; i++)
    {
        if (R[index][i] != 0)
            return false;
    }
    return true;
}
//��R��Ԫ�Ӿ�����û��E�����Ӧ����ʱ����E��Ӧ��������R��Ӧ���г�Ϊ��Ԫ��
void Set_EtoR(unsigned int** R, unsigned int** E, int Eindex, int Rindex)
{
    for (int i = 0; i < R_ColNum; i++)
    {
        R[Rindex][i] = E[Eindex][i];
    }
}
/*����E����Ԫ�о���ĳһ�е�����ֵ*/
void Reset_Efirst(unsigned int** E, int index, int* First)
{
    int i;
    for (i = 0;i < E_ColNum;i++) {
        if (E[index][i] != 0) {
            break;
        }
    }
    if (i == E_ColNum) //����ȫΪ0��û������
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
    First[index] = E_ColNum * 32 - (i + 1) * 32 + j - 1; //�����Ӧ�����ֵ����First��

}
//����Ԫ���������Ӧλ�õ���Ԫ���н������㡪��������
void ExorR(unsigned int** R, unsigned int** E, int Eindex, int Rindex)
{
    for (int i = 0; i < E_ColNum; i++)
    {
        E[Eindex][i] = E[Eindex][i] ^ R[Rindex][i];
    }
}
//�����㷨
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
//������������SSE������
void SSE_ExorR(unsigned int** R, unsigned int** E, int Eindex, int Rindex) {
    int num = 0;
    for (int i = 0; i +4<= E_ColNum; i+=4,num=i)
    {
        __m128i vec_E = _mm_loadu_si128((__m128i*)(E[Eindex] + i));
        __m128i vec_R = _mm_loadu_si128((__m128i*)(R[Rindex] + i));
        vec_E = _mm_xor_si128(vec_E, vec_R);
        _mm_storeu_si128((__m128i*)(E[Eindex]+i), vec_E);
    }
    for (int i = num;i < E_ColNum;i++) {
        E[Eindex][i] = E[Eindex][i] ^ R[Rindex][i];
    }

}
//SSE�㷨
void SSE(int E_RowNum, unsigned int** R, unsigned int** E, int* First)
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
                SSE_ExorR(R, E, i, First[i]);
                Reset_Efirst(E, i, First);
            }
        }
    }
}
//������������AVX������
void AVX_ExorR(unsigned int** R, unsigned int** E, int Eindex, int Rindex) {
    int num = 0;
    for (int i = 0; i + 8 <= E_ColNum; i += 8, num = i)
    {
        __m256i vec_E = _mm256_loadu_si256((__m256i*)(E[Eindex] + i));
        __m256i vec_R = _mm256_loadu_si256((__m256i*)(R[Rindex] + i));
        vec_E = _mm256_xor_si256(vec_E, vec_R);
        _mm256_storeu_si256((__m256i*)(E[Eindex] + i), vec_E);
    }
    for (int i = num;i < E_ColNum;i++) {
        E[Eindex][i] = E[Eindex][i] ^ R[Rindex][i];
    }

}
//AVX�㷨
void AVX(int E_RowNum, unsigned int** R, unsigned int** E, int* First)
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
                AVX_ExorR(R, E, i, First[i]);
                Reset_Efirst(E, i, First);
            }
        }
    }
}
//����Ԫ��������ļ�
void save_result(unsigned int** E, int* First, int E_RowNum)
{
    ofstream outputFile(datapath + "res.txt");
    if (!outputFile) {
        cerr << "Failed to open the Result File��" << endl;
        return;
    }
    for (int i = 0; i < E_RowNum; i++)
    {
        if (First[i] == -1) //��Ӧ��ȫΪ0
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
                if (Bit.test(k)) //�ж�Bit�ĵ� k λ�Ƿ�Ϊ1
                {
                    number = 32 * (E_ColNum - j - 1) + k;
                    //cout << number << " ";
                    outputFile << number << " ";
                }
            }
        }
        //cout << endl;
        outputFile << endl;
    }
    outputFile.close();
}
int main() {
    long long begin, end, freq;
    double timeuse1 = 0, timeuse2 = 0, timeuse3 = 0;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    //���㱻��Ԫ�о��������
    ifstream file(datapath + "2.txt");
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
    //���屻��Ԫ����E������First����Ԫ�о���R
    int* First = new int[E_RowNum];
    unsigned int** E = new unsigned int* [E_RowNum];
    for (int i = 0;i < E_RowNum;i++) {
        E[i] = new unsigned int[E_ColNum];
    }
    unsigned int** R = new unsigned int* [R_RowNum];
    for (int i = 0;i < R_RowNum;i++) {
        R[i] = new unsigned int[R_ColNum];
    }
    //�����㷨��ʱ
    for (int i = 0;i < times;i++) {
        Init_Zero(E, E_RowNum, E_ColNum);
        Init_E(E, First);
        Init_Zero(R, R_RowNum, R_ColNum);
        Init_R(R);
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        serial(E_RowNum, R, E, First);
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timeuse1 += (end - begin) * 1000.0 / freq;
    }
    cout << "col=" << ColNum << "  serial:" << timeuse1 / times << "ms" << endl;
    //SSE�㷨��ʱ
    for (int i = 0;i < times;i++) {
        Init_Zero(E, E_RowNum, E_ColNum);
        Init_E(E, First);
        Init_Zero(R, R_RowNum, R_ColNum);
        Init_R(R);
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        SSE(E_RowNum, R, E, First);
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timeuse2 += (end - begin) * 1000.0 / freq;
    }
    cout << "col=" << ColNum << "  SSE:" << timeuse2 / times << "ms" << endl;
    //AVX�㷨��ʱ
    for (int i = 0;i < times;i++) {
        Init_Zero(E, E_RowNum, E_ColNum);
        Init_E(E, First);
        Init_Zero(R, R_RowNum, R_ColNum);
        Init_R(R);
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        AVX(E_RowNum, R, E, First);
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timeuse3 += (end - begin) * 1000.0 / freq;
    }
    cout << "col=" << ColNum << "  AVX:" << timeuse3 / times << "ms" << endl;
    save_result(E, First, E_RowNum);
    return 0;
}