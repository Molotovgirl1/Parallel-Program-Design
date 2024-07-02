#include<iostream>
#include<fstream>
#include<sstream>
#include<bitset>
#include<cmath>
#include<sys/time.h>
#include<semaphore.h>
#include<cstring>
using namespace std;

const int colNum = 130;
const int R_lineNum = colNum;
const int actual_colNum = ceil(colNum * 1.0 / 32);

int sign = 0;
int eNum;
int* First;
unsigned int** R;
unsigned int** E;
int whichR;


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
bool Is_R_Null(unsigned int** R, int index)
{
    for (int i = 0; i < actual_colNum; i++)
    {
        if (R[index][i] != 0)
            return false;
    }
    return true;
}

/*当R消元子矩阵中没有E首项对应的行时，E的那一行完成消元，同时设置在R对应的行成为消元子*/
void Set_EtoR(unsigned int** R, unsigned int** E, int Eindex, int Rindex)
{
    for (int i = 0; i < actual_colNum; i++)
    {
        R[Rindex][i] = E[Eindex][i];
    }
}


/*重置E被消元行矩阵某一行的首项值*/
void Reset_Efirst(unsigned int** E, int index, int* First)
{
    int i = 0;
    while (E[index][i] == 0 && i < actual_colNum)
    {
        i++;
    }
    if (i == actual_colNum)
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
    First[index] = actual_colNum * 32 - (i + 1) * 32 + j - 1;
}

/*E被消元行减去首项对应位置的R消元子行，本质是异或操作*/
void ExorR(unsigned int** R, unsigned int** E, int Eindex, int Rindex)
{
    for (int i = 0; i < actual_colNum; i++)
    {
        E[Eindex][i] = E[Eindex][i] ^ R[Rindex][i];
    }
}

/*串行算法*/
void serial(int eNum, int* First, unsigned int** R, unsigned int** E)
{
    for (int i = 0; i < eNum; i++)
    {
        while (First[i] != -1)   //当前被消元行是否已完成消元
        {
            if (Is_R_Null(R, First[i]))  //判断是否存在首项对应的消元行
            {
                Set_EtoR(R, E, i, First[i]);  //当前行消元结束,并设置为消元子
                First[i] = -1;  //修改首项为-1表示消元结束
                break;
            }
            else   //如果E当前行首项对应的R的消元子不为空
            {
                ExorR(R, E, i, First[i]);   //消元:被消元行与对应的消元行异或
                Reset_Efirst(E, i, First);   //某行被消元后首项要进行调整
            }
        }
    }
}
void elimination(int rank, int size)
{

    for (int i = rank; i < eNum; i += size)  //当前进程的任务行
    {
        if (!Is_R_Null(R, First[i]))  //首项对应消元子不空
        {
            ExorR(R, E, i, First[i]);   //消元:被消元行与对应的消元行异或
            Reset_Efirst(E, i, First);   //某行被消元后首项要进行调整
        }
    }
    return;
}
void mainfunc_MPI(int eNum, int* First, unsigned int** R, unsigned int** E)
{
    double Tstart, Tend;
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Tstart = MPI_Wtime();
    if (rank == 0) {
        do {
            //0号进程进行任务划分
            for (int i = 0; i < eNum; i++) {
                int flag = i % size;
                if (flag == rank)
                    continue;
                else
                {
                    MPI_Send(&E[i][0], actual_colNum, MPI_UNSIGNED, flag, 0, MPI_COMM_WORLD);   //被消元子行
                    MPI_Send(&First[i], 1, MPI_INT, flag, 0, MPI_COMM_WORLD);        //被消元子行对应的首项
                }
            }

            //处理自己的任务
            elimination(rank, size);

            //0号进程完成自己的任务之后还要接收其他进程完成任务后的结果
            for (int i = 0; i < eNum; i++) {
                int flag = i % size;
                if (flag == rank)
                    continue;
                else
                {
                    MPI_Recv(&E[i][0], actual_colNum, MPI_UNSIGNED, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&First[i], 1, MPI_INT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

            }
            sign = 0;
            //将某个被消元行设置成为消元子，只有0号进程做这一件事
            for (int i = 0; i < eNum; i++) {
                if (First[i] != -1) {    //找到第一个对应消元子行为空的被消元子行
                    if (Is_R_Null(R, First[i])) {
                        Set_EtoR(R, E, i, First[i]);   //将找到的被消元子设置为消元子
                        whichR = First[i];  //记录索引
                        First[i] = -1;   //标记当前行消元结束
                        sign = 1;
                        break;
                    }
                }
            }

            for (int j = 1; j < size; j++) {
                MPI_Send(&whichR, 1, MPI_INT, j, 2, MPI_COMM_WORLD);
                MPI_Send(&sign, 1, MPI_INT, j, 3, MPI_COMM_WORLD);
                MPI_Send(&R[whichR], 1, MPI_INT, j, 4, MPI_COMM_WORLD);
            }

        } while (sign == 1);

    }
    else {

        do {
            //非0号进程先接收任务
            for (int i = rank; i < eNum; i += size) {
                MPI_Recv(&E[i][0], actual_colNum, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&First[i], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            //执行任务
            elimination(rank, size);

            //非0号进程完成任务后将结果传给0号进程
            for (int i = rank; i < eNum; i += size) {
                MPI_Send(&E[i][0], actual_colNum, MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD);
                MPI_Send(&First[i], 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            }
            MPI_Recv(&whichR, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&sign, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&R[whichR], 1, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        } while (sign==1);

    }
    Tend = MPI_Wtime();
    if (rank == 0)
    {
        string str0 = "result0.txt";
        output(E, eNum, actual_colNum, First, str0);
        cout << "colNum=" << colNum << " MPI:" << (Tend - Tstart) * 1000 << "ms" << endl;
    }
}


/*输出消元结果*/
void output(unsigned int** E, int eNum, int col, int* First, string str)
{

    bitset<32> Bit(0);
    std::ofstream ofs(str);
    if (!ofs.is_open())
        cerr << " wrong when output result! " << endl;
    for (int i = 0; i < eNum; i++)
    {
        bool isnull = 1;
        for (int j = 0; j < col; j++) {
            if (E[i][j] != 0) {
                isnull = 0;
                break;
            }
        }
        if (isnull)
        {
            ofs << endl;
            continue;
        }
        std::ostringstream oss;
        for (int j = 0; j < col; j++)
        {
            if (E[i][j] == 0)
                continue;
            Bit = E[i][j];
            for (int k = 31; k >= 0; k--)
            {
                if (Bit.test(k))
                {
                    oss << 32 * (col - j - 1) + k << " ";
                }
            }
        }
        ofs << oss.str() << std::endl;
    }
    ofs.close();
}

int main()
{
    MPI_Init(nullptr, nullptr);
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


    func_mpi();

    MPI_Finalize();

    return 0;

}