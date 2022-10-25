#include <iostream>
#include <algorithm>
#include <climits>
#include <cassert>
#include <stdio.h>
#include <omp.h>
#include <functional>
#include <numeric>

#define d double
using namespace std;


// void naive_seq_scan(d *x, d *y, int n);
void seq_baseline(d *x, d *y, int n);
void par_p_scan_3(d *x, d *y, d *t, int n, int bc);
d par_p_scan_up(d *x, d *t, int i, int j, int bc);
void par_p_scan_down(double v, d *x, d *t, d *y, int i, int j, int bc);
const d double_add(const d x, const d y);

const d double_add(const d x, const d y) {
    return x + y;
}

// void naive_seq_scan(d *x, d *y, int n) {
//     d sum = 0;
//     for (int i = 1; i <= n; ++i) {
//         sum += x[i];
//         y[i] = sum;
//     }
// }

void seq_baseline(d *x, d *y, int n) {
    inclusive_scan(x + 1, x + n + 1, y + 1, double_add, 0.0);
}

void par_p_scan_3(d *x, d *y, d *t, int n, int bc) {
    y[1] = x[1];
    if (n > 1) {
        par_p_scan_up(x, t, 2, n, bc);
        par_p_scan_down(x[1], x, t, y, 2, n, bc);
    }
}

d par_p_scan_up(d *x, d *t, int i, int j, int bc) {
    // if (i == j) return x[i];
    if (j - i < bc) return accumulate(x + i, x + j + 1, 0.0);
    int k = int((i + j) / 2);
    #pragma omp task
    t[k] = par_p_scan_up(x, t, i, k, bc);
    d right = par_p_scan_up(x, t, k + 1, j, bc);
    #pragma omp taskwait
    return t[k] + right;
}

void par_p_scan_down(double v, d *x, d *t, d *y, int i, int j, int bc) {
    // if (i == j) y[i] = v + x[i];
    if (j - i < bc) {
        // printf("par_p_scan_down, i=%d,j=%d\n", i, j);
        inclusive_scan(x + i, x + j + 1, y + i, double_add, v);
    }
    else {
        int k = int((i + j) / 2);
        #pragma omp task
        par_p_scan_down(v, x, t, y, i, k, bc);
        par_p_scan_down(v + t[k], x, t, y, k + 1, j, bc);
        #pragma omp taskwait
    }
}



int main(int argc, char **argv) {
    if (argc != 4) {
        cout << "Usage: ./a.out seed length nthreads" << endl;
        return 1;
    }
    srand(atoi(argv[1]));
    int length = atoi(argv[2]);
    int nthreads = atoi(argv[3]);


    d *x0 = new d[length + 10];  // backup only
    d *x1 = new d[length + 10];
    d *x2 = new d[length + 10];
    d *y1 = new d[length + 10];
    d *y2 = new d[length + 10];
    d *t = new d[length + 10];

    // double test_array[20] = {-1, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    for (int i = 1; i <= length; i++) {
        x0[i] = x1[i] = x2[i] = double(rand() % length) / length;
        // printf("%lf ", x0[i]);
    }

    // d test[10] = {-1,3,2,1,5,7,4,2,1};
    // for (int i = 1; i <= length; i++) {
    //     x0[i] = x1[i] = x2[i] = test[i];//rand() % length;
    // }

    // int bc = 9;

    // omp_set_num_threads(nthreads);
    // #pragma omp parallel
    // {
    //     #pragma omp single
    //     par_p_scan_3(x1, y1, t, length, bc);
    // }
    double start_seq_t = omp_get_wtime();
    seq_baseline(x2, y2, length);
    double seq_t = omp_get_wtime() - start_seq_t;
    

    // printf("div & con:\t");
    // for (int i = 1; i <= length; ++i) {
    //     printf("%.1lf\t", y1[i]);
    // }
    // printf("\n");

    // printf("naive:\t\t");
    // for (int i = 1; i <= length; ++i) {
    //     printf("%.1lf\t", y2[i]);
    // }
    // printf("\n");

    // for(int j = 1; j <= length; j++) {
    //     assert(y1[j] == y2[j]);
    // }

    // printf("Optimal bc in theory is %d / %d = %d\n", length, nthreads, int(length/nthreads));

    // printf("bc\t\ttime_sort\ttime\t\tspeedup\t\tefficiency (%)\n");

    // for (int i=2; i<=16*length/nthreads; i*=2) {
    //     omp_set_num_threads(nthreads);
    //     copy(x0, x0+length+10, x1);
    //     d start_t = omp_get_wtime();
    //     #pragma omp parallel   // [Enze]
    //     {
    //         #pragma omp single  // [Enze]
    //         par_p_scan_3(x1, y1, t, length, i);
    //     }
    //     d elapsed_t = omp_get_wtime() - start_t;
    
    //     for(int j = 1; j <= length; j++) {
    //         if (y1[j] != y2[j]) {
    //             // for (int k=1; k<=length; ++k) {
    //             //     printf("%.1lf\t", y1[i]);
    //             // }
    //             // printf("\n");
    //             // for (int k=1; k<=length; ++k) {
    //             //     printf("%.1lf\t", y2[i]);
    //             // }
    //             // printf("\n");
    //             printf("mismatch at y1[%d] = %lf != y2[%d] = %lf !\n", j, y1[j], j, y2[j]);
    //         }
    //         assert(y1[j] == y2[j]);
    //     }
    //     printf("%d\t\t%.4lf\t\t%.4lf\t\t%.4lf\t\t%.4lf\n", i, seq_t, elapsed_t, seq_t/elapsed_t, seq_t/elapsed_t/nthreads*100.0);
    // }

    printf("p\tbc\t\tseq_t\tpar_t\tspeedup\tefficiency (%)\n");
    // double seq_t;
    for (int i = 1; i <= 44; i++) {
        omp_set_num_threads(i);
        int bc = int(length / i);
        if (bc < 1) {
            bc = 1;
        }
        // bc = 16777216;
        copy(x0, x0 + length + 10, x1);
        double start_t = omp_get_wtime();
        
        #pragma omp parallel
        {
            #pragma omp single
            par_p_scan_3(x1, y1, t, length, bc);
        }
        double elapsed_t = omp_get_wtime() - start_t;
        // if (i == 1) seq_t = elapsed_t;
        for(int j = 1; j <= length; j++) {
            if (y1[j] != y2[j]) {
                printf("[p=%d, bc=%d] mismatch at y1[%d] = %lf != y2[%d] = %lf !\n", i, bc, j, y1[j], j, y2[j]);
                // printf("x:\t");
                // for (int i = 1; i <= length; ++i) {
                //     printf("%.3lf\t", x0[i]);
                // }
                // printf("\n");
                // printf("t:\t");
                // for (int i = 1; i <= length; ++i) {
                //     printf("%.3lf\t", t[i]);
                // }
                // printf("\n");
                // printf("par:\t");
                // for (int i = 1; i <= length; ++i) {
                //     printf("%.3lf\t", y1[i]);
                // }
                // printf("\n");
                // printf("seq:\t");
                // for (int i = 1; i <= length; ++i) {
                //     printf("%.3lf\t", y2[i]);
                // }
                // printf("\n");
            }
            
            assert(y1[j] == y2[j]);
        }
        printf("%d\t%d \t%.4lf\t%.4lf\t%.3lf\t%.3lf\n", i, bc, seq_t, elapsed_t, seq_t/elapsed_t, seq_t/elapsed_t/i*100.0);
    }

    // double test[10] = {1,2,3,4,5,6,7,8,9,10};
    // double dst[10];
    // inclusive_scan(test, test+10, dst, double_add, 0.5);
    // double testsum = accumulate(test, test+10, 0);
    // printf("testsum = %lf\n", testsum);
    // for (int i=0;i<10;i++) printf("%lf ", dst[i]);
    // printf("\n");
    
    delete [] x0;
    delete [] x1;
    delete [] x2;
    delete [] y1;
    delete [] y2;
    delete [] t;
    return 0;
}