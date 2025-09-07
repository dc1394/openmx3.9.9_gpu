#include "openmx_common.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void getNodeName(char** hostname_array, int rank, int size)
{
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hostname_len;

    MPI_Get_processor_name(hostname, &hostname_len);

    // 各プロセスの文字列を収集するための配列
    char* global_strings = NULL;
    int* recv_counts = NULL;
    int* displacements = NULL;

    if (rank == 0) {
        // マスタープロセスがメモリを確保
        global_strings = (char*)malloc(MAX_STRING_LENGTH * size);
        recv_counts = (int*)malloc(size * sizeof(int));
        displacements = (int*)malloc(size * sizeof(int));
    }

    // 各プロセスの文字列の長さを集める
    int local_length = strlen(hostname) + 1;
    MPI_Gather(&local_length, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, mpi_comm_level1);

    // マスタープロセスが受信バッファのディスプレイスメントを計算
    if (rank == 0) {
        displacements[0] = 0;
        for (int i = 1; i < size; i++) {
            displacements[i] = displacements[i - 1] + recv_counts[i - 1];
        }
    }

    // 各プロセスの文字列を集める
    MPI_Gatherv(hostname, local_length, MPI_CHAR,
        global_strings, recv_counts, displacements, MPI_CHAR,
        0, mpi_comm_level1);

    // split
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            strcpy(hostname_array[i], &global_strings[displacements[i]]);
        }
    }

    if (rank == 0) {
        free(global_strings);
        free(recv_counts);
        free(displacements);
    }
}

int getNodeNum(char** hostname_array, int size)
{
    // ユニークな文字列の数を格納する変数
    int uniqueCount = 0;

    // ユニークな文字列を格納する配列
    char** uniqueStrings = (char**)malloc(size * sizeof(char*));
    for (int i = 0; i < size; i++) {
        uniqueStrings[i] = (char*)malloc(MAX_STRING_LENGTH * sizeof(char));
    }

    // 各文字列に対してユニークかどうかを判定
    for (int i = 0; i < size; i++) {
        if (isUnique(hostname_array[i], uniqueStrings, uniqueCount)) {
            strcpy(uniqueStrings[uniqueCount], hostname_array[i]);
            uniqueCount++;
        }
    }

    // メモリの解放
    for (int i = 0; i < size; i++) {
        free(uniqueStrings[i]);
    }
    free(uniqueStrings);

    return uniqueCount;
}

// 文字列がuniqueStringsに既に存在するかどうかを判定する関数
bool isUnique(char* str, char** uniqueStrings, int count)
{
    for (int i = 0; i < count; i++) {
        if (strcmp(str, uniqueStrings[i]) == 0) {
            return false; // 重複が見つかった場合
        }
    }
    return true; // ユニークな場合
}
