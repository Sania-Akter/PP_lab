#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// Function to print a matrix
void display(int rows, int cols, int matrix[rows][cols]) {
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3d ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
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

    // Broadcast to all processes
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Divisible check
    if(K % size != 0) {
        if(rank == 0)
            printf("K must be divisible by number of processes!\n");
        MPI_Finalize();
        return 1;
    }

    int A[K][M][N], B[K][N][P], R[K][M][P];

    // Initialize matrices in root
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

    // Print sample matrix
    if(rank == 0) {
        int idx = 0;
        printf("\nSample Matrix A[0]:\n");
        display(M, N, A[idx]);

        printf("Sample Matrix B[0]:\n");
        display(N, P, B[idx]);

        printf("Result Matrix R[0]:\n");
        display(M, P, R[idx]);
    }

    printf("Process %d Time: %f sec\n", rank, endTime - startTime);

    MPI_Finalize();
    return 0;
}
