#include "crypto_kem.h"

int intended_function = 0;
// int sec_index;
int no_leakage_trials = 0;

int error_now = 0;

int succ_flag = 0;
int count_ones = 0;
int count_plus_ones = 0;
int count_minus_ones = 0;
int non_zero_f_coeff = 0;
int non_zero_g_coeff = 0;
int check_for_value = 0;

int mul_value = 0;

int c_value = 0;
int c_value_1 = 0;
int c_value_2 = 0;
int c_value_for_attack_1 = 0;
int c_value_for_attack_2 = 0;

int c_value_for_leakage = 0;
int collision_index = 0;
int collision_value = 0;
int hw_value = 0;
int weight_hh = 0;

int m = 0;
int n = 0;

small er[p] = {0};
small er_decrypt[p] = {0};
small global_f[p] = {0};
small global_g[p] = {0};

/* ----- arithmetic mod q */

Fq global_c_in_encrypt[p] = {0};
Fq global_c_in_decrypt[p] = {0};
Fq x_f_array[p] = {0};
Fq x_g_array[p] = {0};
Fq cf3[p] = {0};
Fq f_diff_3[p] = {0};

Fq cf[p] = {0};
small e[p] = {0};
small ev[p] = {0};
Fq c_copy[p] = {0};
