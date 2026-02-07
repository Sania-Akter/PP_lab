%%writefile search_phonebook_id.cu
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cuda.h>

using namespace std;

#define MAX_STR_LEN 50

// Device substring check
__device__ bool check(char* str1, char* str2, int len) {
    for(int i = 0; str1[i] != '\0'; i++) {
        int j = 0;
        while(str1[i+j] != '\0' && str2[j] != '\0' && str1[i+j] == str2[j]) {
            j++;
        }
        if(j == len-1) return true;
    }
    return false;
}

// Kernel
__global__ void searchID(char* d_ids,
                         char* d_names,
                         char* d_numbers,
                         int n,
                         char* search,
                         int len) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n) {
        char* id   = d_ids     + idx * MAX_STR_LEN;
        char* name = d_names   + idx * MAX_STR_LEN;
        char* num  = d_numbers + idx * MAX_STR_LEN;

        // ONLY ID CHECK
        if(check(id, search, len)) {
            printf("%s %s %s\n", id, name, num);
        }
    }
}

int main(int argc, char* argv[]) {

    if(argc != 3) {
        cerr << "Usage: " << argv[0] << " <search_id> <threads>\n";
        return 1;
    }

    string search_string = argv[1];
    int num_threads = atoi(argv[2]);
    string file_name = "phonebook.txt";

    // Read file
    vector<string> raw_lines;
    ifstream file(file_name);
    if(!file.is_open()) {
        cerr << "Error opening file\n";
        return 1;
    }

    string line;
    while(getline(file, line)) {
        if(!line.empty()) raw_lines.push_back(line);
    }
    file.close();

    int n = raw_lines.size();

    // Host memory
    char* h_ids     = (char*)malloc(n * MAX_STR_LEN);
    char* h_names   = (char*)malloc(n * MAX_STR_LEN);
    char* h_numbers = (char*)malloc(n * MAX_STR_LEN);

    // Parse lines
    for(int i = 0; i < n; i++) {
        string L = raw_lines[i];

        int p1 = L.find(",");
        int p2 = L.find(",", p1 + 1);

        string id     = L.substr(0, p1);
        string name   = L.substr(p1 + 1, p2 - p1 - 1);
        string number = L.substr(p2 + 1);

        strncpy(h_ids + i*MAX_STR_LEN, id.c_str(), MAX_STR_LEN-1);
        strncpy(h_names + i*MAX_STR_LEN, name.c_str(), MAX_STR_LEN-1);
        strncpy(h_numbers + i*MAX_STR_LEN, number.c_str(), MAX_STR_LEN-1);

        h_ids[i*MAX_STR_LEN + MAX_STR_LEN-1] = '\0';
        h_names[i*MAX_STR_LEN + MAX_STR_LEN-1] = '\0';
        h_numbers[i*MAX_STR_LEN + MAX_STR_LEN-1] = '\0';
    }

    // Device memory
    char *d_ids, *d_names, *d_numbers, *d_search;
    int search_len = search_string.length() + 1;

    cudaMalloc(&d_ids,     n * MAX_STR_LEN);
    cudaMalloc(&d_names,   n * MAX_STR_LEN);
    cudaMalloc(&d_numbers, n * MAX_STR_LEN);
    cudaMalloc(&d_search,  search_len);

    cudaMemcpy(d_ids, h_ids, n * MAX_STR_LEN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_names, h_names, n * MAX_STR_LEN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_numbers, h_numbers, n * MAX_STR_LEN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_search, search_string.c_str(), search_len, cudaMemcpyHostToDevice);

    // Kernel launch
    int blocks = (n + num_threads - 1) / num_threads;

    searchID<<<blocks, num_threads>>>(
        d_ids,
        d_names,
        d_numbers,
        n,
        d_search,
        search_len
    );

    cudaDeviceSynchronize();

    // Cleanup
    free(h_ids);
    free(h_names);
    free(h_numbers);

    cudaFree(d_ids);
    cudaFree(d_names);
    cudaFree(d_numbers);
    cudaFree(d_search);

    return 0;
}
