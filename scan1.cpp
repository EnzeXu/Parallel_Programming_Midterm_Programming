#include <iostream>
#include <algorithm>
#include <climits>
#include <cassert>
#include <stdio.h>
#include <omp.h>

#define d double
using namespace std;

void seq_p_scan_3(d *x, d *y, d *t, int n);
d seq_p_scan_up(d *x, d *t, int i, int j);
void seq_p_scan_down(double v, d *x, d *t, d *y, int i, int j);
void naive_seq_scan(d *x, d *y, int n);
void par_p_scan_3(d *x, d *y, d *t, int n);
d par_p_scan_up(d *x, d *t, int i, int j);
void par_p_scan_down(double v, d *x, d *t, d *y, int i, int j);

void seq_p_scan_3(d *x, d *y, d *t, int n) {
    y[1] = x[1];
    if (n > 1) {
        seq_p_scan_up(x, t, 2, n);
        seq_p_scan_down(x[1], x, t, y, 2, n);
    }
}

d seq_p_scan_up(d *x, d *t, int i, int j) {
    if (i == j) return x[i];
    int k = int((i + j) / 2);
    t[k] = seq_p_scan_up(x, t, i, k);
    d right = seq_p_scan_up(x, t, k + 1, j);
    return t[k] + right;
}

void seq_p_scan_down(double v, d *x, d *t, d *y, int i, int j) {
    if (i == j) y[i] = v + x[i];
    else {
        int k = int((i + j) / 2);
        seq_p_scan_down(v, x, t, y, i, k);
        seq_p_scan_down(v + t[k], x, t, y, k + 1, j);
    }
}

void naive_seq_scan(d *x, d *y, int n) {
    d sum = 0;
    for (int i = 1; i <= n; ++i) {
        sum += x[i];
        y[i] = sum;
    }
}

void par_p_scan_3(d *x, d *y, d *t, int n) {
    y[1] = x[1];
    if (n > 1) {
        par_p_scan_up(x, t, 2, n);
        par_p_scan_down(x[1], x, t, y, 2, n);
    }
}

d par_p_scan_up(d *x, d *t, int i, int j) {
    if (i == j) return x[i];
    int k = int((i + j) / 2);
    #pragma omp task
    t[k] = par_p_scan_up(x, t, i, k);
    d right = par_p_scan_up(x, t, k + 1, j);
    #pragma omp taskwait
    return t[k] + right;
}

void par_p_scan_down(double v, d *x, d *t, d *y, int i, int j) {
    if (i == j) y[i] = v + x[i];
    else {
        int k = int((i + j) / 2);
        #pragma omp task
        par_p_scan_down(v, x, t, y, i, k);
        par_p_scan_down(v + t[k], x, t, y, k + 1, j);
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

    for (int i = 1; i <= length; i++) {
        x0[i] = x1[i] = x2[i] = double(rand() % length) / length;
    }

    // seq_p_scan_3(x1, y1, t, length);
    // naive_seq_scan(x2, y2, length);

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

    // for(int i = 1; i <= length; i++) {
    //     assert(y1[i] == y2[i]);
    // }

    naive_seq_scan(x2, y2, length);

    printf("p\tbc\tseq_t\tpar_t\tspeedup\tefficiency (%)\n");
    double seq_t;
    for (int i = 1; i <= 44; i++) {
        omp_set_num_threads(i);
        copy(x0, x0 + length + 10, x1);
        double start_t = omp_get_wtime();
        
        #pragma omp parallel
        {
            #pragma omp single
            par_p_scan_3(x1, y1, t, length);
        }
        double elapsed_t = omp_get_wtime() - start_t;
        if (i == 1) seq_t = elapsed_t;
        for(int j = 1; j <= length; j++) {
            assert(y1[j] == y2[j]);
        }
        printf("%d\t%d\t%.4lf\t%.4lf\t%.3lf\t%.3lf\n", i, 1, seq_t, elapsed_t, seq_t/elapsed_t, seq_t/elapsed_t/i*100.0);
    }
    
    delete [] x0;
    delete [] x1;
    delete [] x2;
    delete [] y1;
    delete [] y2;
    delete [] t;
    return 0;
}