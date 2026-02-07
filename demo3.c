#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// Function to print matrix into file
void displayFile(FILE *fp, int rows, int cols, int matrix[rows][cols]) {
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(fp, "%3d ", matrix[i][j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int K, M, N, P;

    // -------- INPUT FROM KEYBOARD --------
    if(rank == 0) {
        printf("Enter Number of Matrices: ");
        fflush(stdout);
        scanf("%d", &K);

        printf("Enter Rows of Matrix A: ");
        fflush(stdout);
        scanf("%d", &M);

        printf("Enter Columns of Matrix A: ");
        fflush(stdout);
        scanf("%d", &N);

        printf("Enter Columns of Matrix B: ");
        fflush(stdout);
        scanf("%d", &P);
    }

    // Broadcast
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(K % size != 0) {
        if(rank == 0)
            printf("K must be divisible by number of processes!\n");
        MPI_Finalize();
        return 1;
    }

    int A[K][M][N], B[K][N][P], R[K][M][P];

    // Initialize
    if(rank == 0) {
        srand(time(NULL));
        for(int k = 0; k < K; k++) {
            for(int i = 0; i < M; i++)
                for(int j = 0; j < N; j++)
                    A[k][i][j] = rand() % 10;

            for(int i = 0; i < N; i++)
                for(int j = 0; j < P; j++)
                    B[k][i][j] = rand() % 10;
        }
    }

    int localA[K/size][M][N];
    int localB[K/size][N][P];
    int localR[K/size][M][P];

    MPI_Scatter(A, (K/size)*M*N, MPI_INT, localA,
                (K/size)*M*N, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatter(B, (K/size)*N*P, MPI_INT, localB,
                (K/size)*N*P, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    // Multiplication
    for(int k = 0; k < K/size; k++)
        for(int i = 0; i < M; i++)
            for(int j = 0; j < P; j++) {
                localR[k][i][j] = 0;
                for(int l = 0; l < N; l++)
                    localR[k][i][j] += localA[k][i][l] * localB[k][l][j];
            }

    double endTime = MPI_Wtime();

    MPI_Gather(localR, (K/size)*M*P, MPI_INT,
               R, (K/size)*M*P, MPI_INT, 0, MPI_COMM_WORLD);

    // -------- FILE OUTPUT (Matrix) --------
    if(rank == 0) {
        FILE *fp = fopen("output2.txt", "w");
        if(fp == NULL) {
            printf("File open error!\n");
            MPI_Finalize();
            return 1;
        }

        int idx = 0;

        fprintf(fp, "Sample Matrix A[0]:\n");
        displayFile(fp, M, N, A[idx]);

        fprintf(fp, "Sample Matrix B[0]:\n");
        displayFile(fp, N, P, B[idx]);

        fprintf(fp, "Result Matrix R[0]:\n");
        displayFile(fp, M, P, R[idx]);

        fclose(fp);
    }

    // -------- TERMINAL OUTPUT (TIME) --------
    printf("Process %d Time: %f sec\n", rank, endTime - startTime);

    MPI_Finalize();
    return 0;
}
