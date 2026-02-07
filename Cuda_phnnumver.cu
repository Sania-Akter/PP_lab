%%writefile search_phonebook_number.cu
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
__global__ void searchPhonebook(char* d_names,
                                char* d_numbers,
                                int num_contacts,
                                char* search_number,
                                int number_length) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < num_contacts) {
        char* current_name   = d_names   + (idx * MAX_STR_LEN);
        char* current_number = d_numbers + (idx * MAX_STR_LEN);

        // ONLY NUMBER CHECK
        if(check(current_number, search_number, number_length)) {
            printf("%s %s\n", current_name, current_number);
        }
    }
}

int main(int argc, char* argv[]) {

    if(argc != 3) {
        cerr << "Usage: " << argv[0] << " <phone_number> <threads>\n";
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

    int num_contacts = raw_lines.size();

    // Host memory
    char* h_names   = (char*)malloc(num_contacts * MAX_STR_LEN);
    char* h_numbers = (char*)malloc(num_contacts * MAX_STR_LEN);

    for(int i = 0; i < num_contacts; i++) {
        string current_line = raw_lines[i];
        int pos = current_line.find(",");

        string name   = current_line.substr(0, pos);
        string number = current_line.substr(pos+1);

        strncpy(h_names + i*MAX_STR_LEN, name.c_str(), MAX_STR_LEN-1);
        strncpy(h_numbers + i*MAX_STR_LEN, number.c_str(), MAX_STR_LEN-1);

        h_names[i*MAX_STR_LEN + MAX_STR_LEN-1] = '\0';
        h_numbers[i*MAX_STR_LEN + MAX_STR_LEN-1] = '\0';
    }

    // Device memory
    char *d_names, *d_numbers, *d_search;
    int search_len = search_string.length() + 1;

    cudaMalloc(&d_names,   num_contacts * MAX_STR_LEN);
    cudaMalloc(&d_numbers, num_contacts * MAX_STR_LEN);
    cudaMalloc(&d_search,  search_len);

    cudaMemcpy(d_names, h_names,
               num_contacts * MAX_STR_LEN,
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_numbers, h_numbers,
               num_contacts * MAX_STR_LEN,
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_search, search_string.c_str(),
               search_len,
               cudaMemcpyHostToDevice);

    // Launch kernel
    int blocks = (num_contacts + num_threads - 1) / num_threads;

    searchPhonebook<<<blocks, num_threads>>>(
        d_names,
        d_numbers,
        num_contacts,
        d_search,
        search_len
    );

    cudaDeviceSynchronize();

    // Cleanup
    free(h_names);
    free(h_numbers);
    cudaFree(d_names);
    cudaFree(d_numbers);
    cudaFree(d_search);

    return 0;
}
