#include <omp.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>

using std::chrono::high_resolution_clock;
using std::chrono::duration;


// Used for accessing flattend arrays
int idx(int n, int m) {
    return (n*(n + 1)/2.0) + m;
}

// Printing Function for debugging
void printCoefficients(const double* arr, int maxDegree, int count) {
    int printed = 0;
    for (int n = 0; n <= maxDegree && printed < count; ++n) {
        for (int m = 0; m <= n && printed < count; ++m) {
            int i = idx(n, m);
            std::cout << "arr[" << n << "," << m << "] = " << arr[i] << "\n";
            printed++;
        }
    }
}

void sum_harmonics(double* sum, const double* cosArray, const double* sinArray, int maxDegreeOfSum, const double* latGrid, const double* lonGrid, int gridSize) {

    // sum over all orders
    #pragma omp parallel for schedule(dynamic)
    for(int n = 0; n <= maxDegreeOfSum; n++) {
        high_resolution_clock::time_point start;
        high_resolution_clock::time_point end;
        duration<double, std::milli> duration_sec;
        start = high_resolution_clock::now();
        for(int m = 0; m < n + 1; m++) {
            double* temp = new double[gridSize];
            double theta;
            double phi;
            double exp_mPhi;
            double coef;
            int index1;
            int index2;

            // Populate Arrays
            for(int i = 0; i < gridSize; i++){
                theta = latGrid[i] * M_PI / 180.0;
                phi = lonGrid[i] * M_PI / 180.0;
                coef = (cosArray[idx(n,m)]*std::cos(m*phi) + sinArray[idx(n,m)]*std::sin(m*phi));
                temp[i] = std::sph_legendre(n, m, theta) * coef;
                #pragma omp atomic
                sum[i] += temp[i];
            }

            // // Copy over values
            // for(int i = 0; i < gridSize; i++) {

            // }

            delete[] temp;
        }

        // Progress printout
        end = high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
        std::cout << "===========================\n";
        std::cout << "Step " << n << "/" << maxDegreeOfSum << ": " << duration_sec.count() << "ms\n";
    }

    // #pragma omp parallel
    // {
    //     // 1. Each thread gets its own local_sum array initialized to zero:
    //     double* local_sum = new double[gridSize](); // () zero-initializes the array

    //     // 2. Distribute the outer loop (over n) across threads dynamically:
    //     #pragma omp for schedule(dynamic)
    //     for (int n = 0; n < maxDegreeOfSum; n++) {
    //         for (int m = 0; m < n + 1; m++) {
    //             for (int i = 0; i < gridSize; i++) {
    //                 // Calculate theta, phi, coefficient and add to this thread’s local sum
    //                 double theta = latGrid[i] * M_PI / 180.0;
    //                 double phi = lonGrid[i] * M_PI / 180.0;
    //                 double coef = (cosArray[idx(n, m)] * std::cos(m * phi) +
    //                             sinArray[idx(n, m)] * std::sin(m * phi));
    //                 local_sum[i] += std::sph_legendre(n, m, theta) * coef;
    //             }
    //         }
    //     }

    //     // 3. After finishing all assigned n’s, each thread adds its local_sum to the global sum:
    //     #pragma omp critical
    //     {
    //         for (int i = 0; i < gridSize; i++) {
    //             sum[i] += local_sum[i];
    //         }
    //     }

    //     // 4. Clean up the local sum array
    //     delete[] local_sum;
    // }


}


int check_degree(std::string filename) {
    std::ifstream infile(filename, std::ios::binary);

    if (!infile) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return 1;
    }

    // Check that first two values are 0 and max degree
    double first = 0.0, second = 0.0;
    infile.read(reinterpret_cast<char*>(&first), sizeof(double));
    infile.read(reinterpret_cast<char*>(&second), sizeof(double));
    // if (infile) {
    //     std::cout << "First value: " << first << "\n";
    //     std::cout << "Second value: " << second << "\n";
    // } else {
    //     std::cerr << "Failed to read two doubles from the file." << std::endl;
    // }

    return second;

}


int read_bshc(std::string filename, double* cosArray, double* sinArray) {
    int maxDegree = check_degree(filename);

    std::ifstream infile(filename, std::ios::binary);


    // Skip first two inputs
    infile.seekg(2 * sizeof(double), std::ios::beg);

    // Populate Cosine array
    for (int n = 0; n <= maxDegree; ++n) {
        for (int m = 0; m <= n; ++m) {
            infile.read(reinterpret_cast<char*>(&cosArray[idx(n,m)]), sizeof(double));
        }
    }

    // Populate Sine Array
    for (int n = 0; n <= maxDegree; ++n) {
        for (int m = 0; m <= n; ++m) {
            infile.read(reinterpret_cast<char*>(&sinArray[idx(n,m)]), sizeof(double));
        }
    }

    return 0;
}


void create_mesh_grid(const double* latArray, const double* lonArray, double* latGrid, double* lonGrid, int nLat, int nLon) {
    for (int i = 0; i < nLat; ++i) {
        for (int j = 0; j < nLon; ++j) {
            latGrid[i * nLon + j] = latArray[i];
            lonGrid[i * nLon + j] = lonArray[j];
        }
    }
}

void linspace(double* arr, int N, double minVal, double maxVal) {
    for(int i = 0; i < N; i++) {
        arr[i] = minVal + (maxVal - minVal)*(i/((double)(N-1)));
    }
}

int write_to_csv(const double* array, std::string filename, int nRows, int nCols) {
    std::ofstream outfile(filename);

    if (!outfile.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << "\n";
        return 1;
    }

    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            outfile << array[i * nCols + j];
            if (j < nCols - 1) {
                outfile << ",";
            }
        }
        outfile << "\n";
    }

    outfile.close();

    return 0;
}

int main(int argc, char *argv[]) {
    // high_resolution_clock::time_point start;
    // high_resolution_clock::time_point end;
    // duration<double, std::milli> duration_sec;
    // start = high_resolution_clock::now();

    // Check if correct number of arguments are provided
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <filename> <max degree of sum> <scale factor>" << std::endl;
        return 1;
    }

    // Command Line Arguments
    std::string filename = argv[1];
    int maxDegreeOfSum = std::stoi(argv[2]);
    int scaleFactor = std::stoi(argv[3]);
    std::string outfilename = argv[4];

    // Check degree of file Data
    int maxDegree = check_degree(filename);

    if (maxDegreeOfSum < 0 || maxDegreeOfSum > maxDegree) {
        std::cerr << "WARNING: Requested sum order " << maxDegreeOfSum << " invalid, using max degree of dataset (" << maxDegree << ") instead.\n";
        maxDegreeOfSum = maxDegree;
    }

    // Initialize Data Arrays
    int numTerms = (maxDegree + 1) * (maxDegree + 2) / 2;
    double* C = new double[numTerms];
    double* S = new double[numTerms];

    // Populate Data
    read_bshc(filename, C, S);

    // Create Grid
    int minLat = 0;
    int maxLat = 180;
    int minLon = 0;
    int maxLon = 360;
    int latSize = scaleFactor * (maxLat - minLat);
    int lonSize = scaleFactor * (maxLon - minLon);
    int gridSize = latSize * lonSize;
    double* latArray = new double[latSize];
    double* lonArray = new double[lonSize];
    linspace(latArray, latSize+1, minLat, maxLat);
    linspace(lonArray, lonSize+1, minLon, maxLon);
    double* latGrid  = new double[gridSize];
    double* lonGrid  = new double[gridSize];
    create_mesh_grid(latArray, lonArray, latGrid, lonGrid, latSize, lonSize);

    // Sum over array
    double* result = new double[gridSize];
    sum_harmonics(result, C, S, maxDegreeOfSum, latGrid, lonGrid, gridSize);

    // print files
    write_to_csv(result, outfilename, latSize, lonSize);
    // write_to_csv(latGrid, "out/lats.csv", latSize, lonSize);
    // write_to_csv(lonGrid, "out/longs.csv", latSize, lonSize);
    // write_to_csv(latArray, "out/latArr.csv", latSize, 1);
    // write_to_csv(lonArray, "out/longArr.csv", 1, lonSize);

    // free memory
    delete[] C;
    delete[] S;
    delete[] latArray;
    delete[] lonArray;
    delete[] latGrid;
    delete[] lonGrid;
    delete[] result;

    // // Print Total Time
    // end = high_resolution_clock::now();
    // duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
    // std::cout << "time: " << duration_sec.count() << "ms\n";

    return 0;
}