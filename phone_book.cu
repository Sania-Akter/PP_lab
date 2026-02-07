%%writefile search_phonebook.cu
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <cuda.h>

using namespace std;

#define MAX_STR_LEN 50

// CUDA device function to perform substring check
__device__ bool check(char* str1, char* str2, int len) {
    for(int i = 0; str1[i] != '\0'; i++) {
        int j = 0;
        while(str1[i+j] != '\0' && str2[j] != '\0' && str1[i+j] == str2[j]) {
            j++;
        }
        if(j == len-1) {
            return true;
        }
    }
    return false;
}

// Kernel
__global__ void searchPhonebook(char* d_names, char* d_numbers,
                                int num_contacts,
                                char* search_name,
                                int name_length) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < num_contacts) {
        char* current_name   = d_names   + (idx * MAX_STR_LEN);
        char* current_number = d_numbers + (idx * MAX_STR_LEN);

        if(check(current_name, search_name, name_length)) {
            printf("%s %s\n", current_name, current_number);
        }
    }
}

int main(int argc, char* argv[]) {

    if(argc != 3) {
        cerr << "Usage: " << argv[0] << " <search_name> <threads>\n";
        return 1;
    }

    string search_string = argv[1];
    int num_threads = atoi(argv[2]);
    string file_name = "/content/sample_data/phonebook.txt";

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

    int num_contacts = raw_lines.size();

    // Host memory
    char* h_names   = (char*)malloc(num_contacts * MAX_STR_LEN * sizeof(char));
    char* h_numbers = (char*)malloc(num_contacts * MAX_STR_LEN * sizeof(char));

    for(int i = 0; i < num_contacts; i++) {
        string current_line = raw_lines[i];
        int pos = current_line.find(",");

        string name   = current_line.substr(1, pos - 2);
        string number = current_line.substr(pos + 2,
                         current_line.size() - pos - 3);

        strncpy(h_names + i*MAX_STR_LEN, name.c_str(), MAX_STR_LEN-1);
        strncpy(h_numbers + i*MAX_STR_LEN, number.c_str(), MAX_STR_LEN-1);

        h_names[i*MAX_STR_LEN + MAX_STR_LEN-1] = '\0';
        h_numbers[i*MAX_STR_LEN + MAX_STR_LEN-1] = '\0';
    }

    // Device memory
    char *d_names, *d_numbers, *d_search_name;
    int name_len = search_string.length() + 1;

    cudaMalloc(&d_names,   num_contacts * MAX_STR_LEN * sizeof(char));
    cudaMalloc(&d_numbers, num_contacts * MAX_STR_LEN * sizeof(char));
    cudaMalloc(&d_search_name, name_len * sizeof(char));

    cudaMemcpy(d_names, h_names,
               num_contacts * MAX_STR_LEN * sizeof(char),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_numbers, h_numbers,
               num_contacts * MAX_STR_LEN * sizeof(char),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_search_name, search_string.c_str(),
               name_len * sizeof(char),
               cudaMemcpyHostToDevice);

    // Launch kernel in chunks
    for(int i = 0; i < num_contacts; i += num_threads) {
        int thread_count = min(num_contacts - i, num_threads);

        searchPhonebook<<<1, thread_count>>>(
            d_names   + i*MAX_STR_LEN,
            d_numbers + i*MAX_STR_LEN,
            thread_count,
            d_search_name,
            name_len
        );
        cudaDeviceSynchronize();
    }

    // Cleanup
    free(h_names);
    free(h_numbers);
    cudaFree(d_names);
    cudaFree(d_numbers);
    cudaFree(d_search_name);

    return 0;
}
