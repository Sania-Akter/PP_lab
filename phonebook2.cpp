#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

// -------- ID MATCH --------
bool match_id(const string &line, const string &search_id) {
    istringstream iss(line);
    string id;
    getline(iss, id, ',');   // first field = ID
    return id == search_id;
}

// Send string
void send_string(const string &text, int receiver) {
    int len = text.size() + 1;
    MPI_Send(&len, 1, MPI_INT, receiver, 1, MPI_COMM_WORLD);
    MPI_Send(text.c_str(), len, MPI_CHAR, receiver, 1, MPI_COMM_WORLD);
}

// Receive string
string receive_string(int sender) {
    int len;
    MPI_Status status;
    MPI_Recv(&len, 1, MPI_INT, sender, 1, MPI_COMM_WORLD, &status);
    char *buf = new char[len];
    MPI_Recv(buf, len, MPI_CHAR, sender, 1, MPI_COMM_WORLD, &status);
    string res(buf);
    delete[] buf;
    return res;
}

// String → Vector
vector<string> string_to_vector(const string &text) {
    vector<string> lines;
    istringstream iss(text);
    string line;
    while (getline(iss, line))
        if (!line.empty()) lines.push_back(line);
    return lines;
}

// Vector → String
string vector_to_string(const vector<string> &lines, int start, int end) {
    string result;
    for (int i = start; i < min((int)lines.size(), end); i++)
        result += lines[i] + "\n";
    return result;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            cout << "Usage: mpirun -n 4 ./a.out data.txt <ID>\n";
        MPI_Finalize();
        return 1;
    }

    string search_id = argv[argc - 1];
    double start, end;

    if (rank == 0) {
        // -------- MASTER --------
        ifstream f(argv[1]);
        vector<string> lines;
        string line;
        int id_counter = 1;

        // AUTO ID ADD
        while (getline(f, line)) {
            if (!line.empty()) {
                lines.push_back(to_string(id_counter) + "," + line);
                id_counter++;
            }
        }

        int total = lines.size();
        int chunk = (total + size - 1) / size;

        // Send chunks
        for (int i = 1; i < size; i++) {
            send_string(vector_to_string(lines, i * chunk, (i + 1) * chunk), i);
        }

        start = MPI_Wtime();
        vector<string> matches;

        // Master own search
        for (int i = 0; i < min(chunk, total); i++)
            if (match_id(lines[i], search_id))
                matches.push_back(lines[i]);

        // Receive worker results
        for (int i = 1; i < size; i++) {
            string recv = receive_string(i);
            vector<string> v = string_to_vector(recv);
            matches.insert(matches.end(), v.begin(), v.end());
        }

        end = MPI_Wtime();

        // Write output
        ofstream out("output.txt");
        for (auto &m : matches) out << m << "\n";
        out.close();

        cout << "Found " << matches.size() << " match(es)\n";
        cout << "Time: " << end - start << " sec\n";
    }
    else {
        // -------- WORKER --------
        string recv = receive_string(0);
        vector<string> local = string_to_vector(recv);

        start = MPI_Wtime();
        string res = "";

        for (auto &l : local)
            if (match_id(l, search_id))
                res += l + "\n";

        end = MPI_Wtime();

        send_string(res, 0);
        printf("Process %d time %f sec\n", rank, end - start);
    }

    MPI_Finalize();
    return 0;
}
