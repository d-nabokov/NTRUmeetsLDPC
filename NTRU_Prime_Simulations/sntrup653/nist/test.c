/*
   PQCgenKAT_kem.c
   Created by Bassham, Lawrence E (Fed) on 8/29/17.
   Copyright © 2017 Bassham, Lawrence E (Fed). All rights reserved.
   + mods from djb: see KATNOTES
*/
#include <assert.h>  
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rng.h"
#include "crypto_kem.h"
#include "attack_parameters.h"
#include "math.h"
#include "int8.h"
#include "int16.h"
#include "int32.h"
#include "uint16.h"
#include "uint32.h"
#include <math.h>
#include <limits.h>  
#include <stdbool.h> 
#include <float.h>
#include <time.h>
#include "crypto_kem_sntrup653.h"  
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wparentheses-equality"
#pragma GCC diagnostic ignored "-Wimplicit-int"
#pragma GCC diagnostic ignored "-Wunused-label"

extern int no_true_collisions;

#define min(a,b) (((a) < (b)) ? (a) : (b))
double mean_now = 0;
double std_dev = 2;

#define MAX_COMMAND_LENGTH 1024
#define MAX_OUTPUT_SIZE 653  
#define OUTPUT_WEIGHT_FILENAME "/home/fei/C/SCA_Assisted_CCA_on_NTRU/NTRU_Prime/PC_Oracle_based_SCA/Attack_Simulations/sntrup761/SCA-LDPC/simulate-with-python/fprime_weight.txt" 
#define OUTPUT_LDPC_PATH "/home/fei/C/SCA_Assisted_CCA_on_NTRU/NTRU_Prime/PC_Oracle_based_SCA/Attack_Simulations/sntrup761/SCA-LDPC/simulate-with-python/outfile.txt"

#define MAX_SECRET 9 // For secret values from -2 to 2
#define MAX_BIT_LENGTH 10

#define MAX_EXPR_LENGTH 256   

#define N_CODEWORDS 3
#define max_x 3 

#define NUM_COLS 2    

// The path where you want to store output_matrix.txt
#define OUTPUT_MATRIX_PATH "/home/fei/C/SCA_Assisted_CCA_on_NTRU/NTRU_Prime/PC_Oracle_based_SCA/Attack_Simulations/sntrup761/SCA-LDPC/simulate-with-python/output_matrix.txt"
#define OUTPUT_FPRIME_PATH "/home/fei/C/SCA_Assisted_CCA_on_NTRU/NTRU_Prime/PC_Oracle_based_SCA/Attack_Simulations/sntrup761/SCA-LDPC/simulate-with-python/fprime_output.txt"

int **sec_index_second_part;  
int **backup_sec_index_second_part;  


void load_ldpc_from_file(int **sec_index_second_part, int *u_1_u_2, int max_rows) {
    FILE *fp;
    int row = 0;

    fp = fopen(OUTPUT_MATRIX_PATH, "r");
    if (fp == NULL) {
        perror("Error opening output_matrix.txt");
        exit(1);
    }

    printf("File opened successfully. Reading data...\n");

    while (fscanf(fp, "%d,%d", &sec_index_second_part[row][0], &sec_index_second_part[row][1]) == 2) {
        row++;
        if (row >= max_rows) {  
            printf("Maximum rows reached: %d\n", max_rows);
            break;
        }
    }

    *u_1_u_2 = row;  
    fclose(fp);
    printf("File closed successfully.\n");
}


static int shift_lfsr(unsigned int *lfsr, unsigned int polynomial_mask)
{
    int feedback;

    feedback = *lfsr & 1;
    *lfsr >>= 1;
    if(feedback == 1)
        *lfsr ^= polynomial_mask;
    return *lfsr;
}

static int get_random(void)
{
    int temp;
    unsigned int POLY_MASK_HERE_1 = 0x12431212;
    unsigned int POLY_MASK_HERE_2 = 0xABBBEECD;
    static unsigned int lfsr_1 = 0x55AAEEFF;
    static unsigned int lfsr_2 = 0xFFAA8844;
    shift_lfsr(&lfsr_1, POLY_MASK_HERE_1);
    shift_lfsr(&lfsr_2, POLY_MASK_HERE_2);
    temp = (shift_lfsr(&lfsr_1, POLY_MASK_HERE_1) ^ shift_lfsr(&lfsr_2, POLY_MASK_HERE_2)) & 0XFF;
    return (temp);
}



/* x must not be close to top int32 */
static Fq Fq_freeze(int32 x)
{
  return int32_mod_uint14(x+q12,q)-q12;
}

#define KAT_SUCCESS          0
#define KAT_FILE_OPEN_ERROR -1
#define KAT_CRYPTO_FAILURE  -4
//#define NO_TESTS 1

//----- Defines -------------------------------------------------------------
#define PI         3.14159265   // The value of pi

//----- Function prototypes -------------------------------------------------
double norm(double mean, double std_dev);  // Returns a normal rv
double rand_val(int seed);                 // Jain's RNG

//===========================================================================
//=  Function to generate normally distributed random variable using the    =
//=  Box-Muller method                                                      =
//=    - Input: mean and standard deviation                                 =
//=    - Output: Returns with normally distributed random variable          =
//===========================================================================
double norm(double mean, double std_dev)
{
  double   u, r, theta;           // Variables for Box-Muller method
  double   x;                     // Normal(0, 1) rv
  double   norm_rv;               // The adjusted normal rv

  // Generate u
  u = 0.0;
  while (u == 0.0)
    u = ((double)(get_random()*256 + get_random()))/65536;

  // Compute r
  r = sqrt(-2.0 * log(u));

  // Generate theta
  theta = 0.0;
  while (theta == 0.0)
    theta = 2.0 * PI * ((double)(get_random()*256 + get_random()))/65536;

  // Generate x value
  x = r * cos(theta);
  // printf("u: %f, x: %f, r: %f\n, theta: %d\n", u, x, r, theta);

  // Adjust x value for specified mean and variance
  norm_rv = (x * std_dev) + mean;

  // Return the normally distributed RV value
  return(norm_rv);
}

//=========================================================================
//= Multiplicative LCG for generating uniform(0.0, 1.0) random numbers    =
//=   - x_n = 7^5*x_(n-1)mod(2^31 - 1)                                    =
//=   - With x seeded to 1 the 10000th x value should be 1043618065       =
//=   - From R. Jain, "The Art of Computer Systems Performance Analysis," =
//=     John Wiley & Sons, 1991. (Page 443, Figure 26.2)                  =
//=========================================================================
double rand_val(int seed)
{
  const long  a =      16807;  // Multiplier
  const long  m = 2147483647;  // Modulus
  const long  q_now =     127773;  // m div a
  const long  r =       2836;  // m mod a
  static long x;               // Random int value
  long        x_div_q;         // x divided by q
  long        x_mod_q;         // x modulo q
  long        x_new;           // New x value

  // Set the seed if argument is non-zero and then return zero
  if (seed > 0)
  {
    x = seed;
    return(0.0);
  }

  // RNG using integer arithmetic
  x_div_q = x / q_now;
  x_mod_q = x % q_now;
  x_new = (a * x_mod_q) - (r * x_div_q);
  if (x_new > 0)
    x = x_new;
  else
    x = x_new + m;

  // Return a random value between 0.0 and 1.0
  return((double) x / m);
}

// void	fprintBstr(FILE *fp, char *S, unsigned char *A, unsigned long long L);

static uint8_t hw_calc(int8_t byte)
{
  uint8_t bit;
  uint8_t weight = 0;
  for(int i = 0; i < 8; i++)
  {
    bit = (byte >> i)&0x1;
    weight = weight+bit;
  }
  return weight;
}


#define u_row_list_size p
int u_column_size;
int *sec_index; 


int weight_list_k1 = 200;
int weight_list_k2 = 300;
int u_1_u_2;
int collision_count;


#define table_row_size 5 
#define table_column_size 4  
#define new_table_row_size 9


// beta_u\in [-3,3], distingush 1
int c_value_for_attack_list_1[3][3] = 
{   
    {90,288,30}, //(48,42) 
    {96,282,30}, //(48,42) 
    {102,276,30}, //(48,42) 
};

//  distingush 2
int c_value_for_attack_list_2[3][3] = 
{   
    {84,270,42}, //(66,60) 
    {87,264,45}, //(66,60) 
    {99,255,42}, //(66,60) 
};



int param_list1_index;
int param_list2_index;
int param_which_list;
int Beta_which;

int column[9][1] = {0};
int column_2[4][1] = {0}; 
double distr[MAX_SECRET]; // Array to simulate the distribution for -2 to 2

int actual_match_table[table_row_size][table_column_size] = {0}; 


int await_match_table[9][10] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, -1}, //-4
    {0, 0, 0, 0, 0, 0, 0, 0, -1, -1}, //-3
    {0, 0, 0, 0, 0, 0, 0, -1, -1, -1}, //-2
    {0, 0, 0, 0, 0, 0, -1, -1, -1, -1}, //-1
    {0, 0, 0, 0, 0, -1, -1, -1, -1, -1}, //0
    {0, 0, 0, 0, -1, -1, -1, -1, -1, -1}, //1
    {0, 0, 0, -1, -1, -1, -1, -1, -1, -1}, //2
    {0, 0, -1, -1, -1, -1, -1, -1, -1, -1}, //3
    {0, -1, -1, -1, -1, -1, -1, -1, -1, -1}, //4
};

int await_match_table_2[2][3] = {
    {0, -1, -1}, 
    {0, 0, -1},
};


double average_match_table[9][1] = {
    {0.07594321797705739}, // -4
    {1.1194349854005328}, // -3
    {12.468133086274028},  // -2
    {48.39533877749857},  // -1
    {75.88229986569961},  // 0
    {48.39533877749857},  // 1
    {12.468133086274028},  // 2
    {1.1194349854005328},  // 3
    {0.07594321797705739}  // 4
};



typedef struct {
    int param_which_list;
    int param_list1_index;
    int param_list2_index;
} ParamRecord;


void calculate_counts(int* array, int size, double result[1][2]) {
    
    result[0][0] = 0; 
    result[0][1] = 0; 

    for (int i = 0; i < size; i++) {
        if (array[i] == 0) {
            result[0][0]++;
        } else if (array[i] == -1) {
            result[0][1]++;
        }
    }
}

// Compute distance function
double calculate_distance(double expected_zeros, double expected_minus_ones, double actual_zeros, double actual_minus_ones) {
    double distance = (expected_zeros - actual_zeros) * (expected_zeros - actual_zeros) +
                      (expected_minus_ones - actual_minus_ones) * (expected_minus_ones - actual_minus_ones);
    printf("Calculated distance: %f (expected_zeros: %f, actual_zeros: %f, expected_minus_ones: %f, actual_minus_ones: %f)\n",
           distance, expected_zeros, actual_zeros, expected_minus_ones, actual_minus_ones);
    return distance;
}


void update_match_table(double counts[1][2], double average_match_table[9][1]) 
{
    for (int col = 0; col < 1; col++) 
    { 
        double min_distance = DBL_MAX;
        int best_y_index = 0;

        for (int y_index = 5; y_index > -5; y_index--) 
        {
            double expected_zeros = 0.0;
            double expected_minus_ones = 0.0;

            int column_index = 5 - y_index;

            for (int row = 0; row < 9; row++) 
            {
                if (await_match_table[row][column_index] == 0) 
                {
                    expected_zeros += average_match_table[row][0];
                    //  expected_zeros += average_match_table[row][0] + average_match_table[table_row_size-1-row][0];
                } else if (await_match_table[row][column_index] == -1) 
                {
                    expected_minus_ones += average_match_table[row][0];
                    // expected_minus_ones += average_match_table[row][0] + average_match_table[table_row_size-1-row][0];
                }
            }
          
           printf("For y_index = %d, expected_zeros = %f, expected_minus_ones = %f\n", y_index, expected_zeros, expected_minus_ones);

           
            double actual_zeros = counts[col][0];
            double actual_minus_ones = counts[col][1];

            double distance = calculate_distance(expected_zeros, expected_minus_ones, actual_zeros, actual_minus_ones);

            if (distance < min_distance) {
                min_distance = distance;
                best_y_index = y_index;
            }
        }

        int best_column_index = 5 - best_y_index;

        for (int row = 0; row < 9; row++) {
            column[row][col] = await_match_table[row][best_column_index];
        }

        printf("Determined column for col %d (column format):\n", col);
        for (int row = 0; row < 9; row++) {
            printf("%d\n", column[row][col]);
        }
    }
}


void update_match_table_2(double counts[1][2], int new_array_size) {
    for (int col = 0; col < 1; col++) { 
        double min_distance = DBL_MAX;
        int best_y_index = 0;

        for (int y_index = 0; y_index < 3; y_index++) {
            double expected_zeros = 0.0;
            double expected_minus_ones = 0.0;

            if (y_index == 0) {
                expected_zeros = new_array_size * (1 - 0);
                expected_minus_ones = new_array_size * 0;
            } else if (y_index == 1) {
                expected_zeros = new_array_size * (1 - 0.22017022994894594);
                expected_minus_ones = new_array_size * 0.22017022994894594;
            } else if (y_index == 2) {
                expected_zeros = new_array_size * 0;
                expected_minus_ones = new_array_size * 1;
            }

            printf("For y_index = %d, expected_zeros = %f, expected_minus_ones = %f\n", y_index, expected_zeros, expected_minus_ones);

            double actual_zeros = counts[col][0];
            double actual_minus_ones = counts[col][1];

            double distance = calculate_distance(expected_zeros, expected_minus_ones, actual_zeros, actual_minus_ones);

            if (distance < min_distance) {
                min_distance = distance;
                best_y_index = y_index;
            }
        }

        int best_column_index = best_y_index;

        for (int row = 0; row < 2; row++) {
            column_2[row][col] = await_match_table_2[row][best_column_index];
        }

        printf("Determined column for col %d (column format):\n", col);
        for (int row = 0; row < 2; row++) {
            printf("%d\n", column_2[row][col]);
        }
    }
}


void calculate_beta_probabilities(long double *Pr, int beta_coeff_count) {
    for (int i = 0; i < 9; i++) {
        Pr[i] = 0.0L;
    }

    if (beta_coeff_count == 1) {
        Pr[3] = 144.0L / 653.0L;
        Pr[4] = 365.0L / 653.0L;
        Pr[5] = 144.0L / 653.0L;
    } else if (beta_coeff_count == 2) {
        Pr[2] = 20736.0L / 426409.0L;
        Pr[3] = 105120.0L / 426409.0L;
        Pr[4] = 174697.0L / 426409.0L;
        Pr[5] = 105120.0L / 426409.0L;
        Pr[6] = 20736.0L / 426409.0L;
    } else if (beta_coeff_count == 3) {
        Pr[1] = 2985984.0L / 278445077.0L;
        Pr[2] = 22705920.0L / 278445077.0L;
        Pr[3] = 66511152.0L / 278445077.0L;
        Pr[4] = 94038965.0L / 278445077.0L;
        Pr[5] = 66511152.0L / 278445077.0L;
        Pr[6] = 22705920.0L / 278445077.0L;
        Pr[7] = 2985984.0L / 278445077.0L;
    } else if (beta_coeff_count == 4) {
        Pr[0] = 429981696.0L / 181824635281.0L;
        Pr[1] = 4359536640.0L / 181824635281.0L;
        Pr[2] = 18295248384.0L / 181824635281.0L;
        Pr[3] = 41087833920.0L / 181824635281.0L;
        Pr[4] = 53479434001.0L / 181824635281.0L;
        Pr[5] = 41087833920.0L / 181824635281.0L;
        Pr[6] = 18295248384.0L / 181824635281.0L;
        Pr[7] = 4359536640.0L / 181824635281.0L;
        Pr[8] = 429981696.0L / 181824635281.0L;
    }
}


void generate_base_ciphertext(int *success_trial, unsigned char *ct, unsigned char *ss, unsigned char *pk, unsigned char *sk, int param_p, int *no_single_collisions, int *no_multiple_collisions, int *no_false_negative_collisions, int *no_false_positive_collisions, int kem_CIPHERTEXTBYTES, char *ct_file_now_basic, char *ct_file_basic_failed, int *collision_array_index, int *collision_array_value, int *collision_count,int *total_profile_trials_overall, int *false_positive_count, int c_base_vote_limit, int *overall_oracle_count) {


    int votes[p]; 
    while (*success_trial == 0) 
    {
        int got_minus_one = 0;
        int got_zero = 0;
        int count_non_zero_coeffs = 0; 
        *collision_count = 0; 
        // *false_positive_count=0;
        crypto_kem_enc(ct, ss, pk);
        crypto_kem_dec(ss, ct, sk);

        for (int i = 0; i < param_p; i++) {
            if (er_decrypt[i] == 1 || er_decrypt[i] == -1) {
                (*collision_count)++;
            }
        }
        for (int vote = 0; vote < c_base_vote_limit; vote++) 
        {           
            count_non_zero_coeffs = 0;            
            (*total_profile_trials_overall)++;    
            (*overall_oracle_count)++;

            for (int i = 0; i < param_p; i++) {
                if (er_decrypt[i] == 1 || er_decrypt[i] == -1) {
                    count_non_zero_coeffs++;
                }
            }

            // Introduce Bernoulli noise
            double noise_probability = 0.96;
            unsigned char random_value_bytes[4];
            randombytes(random_value_bytes, sizeof(random_value_bytes));  

            uint32_t random_value_int = (random_value_bytes[0] | (random_value_bytes[1] << 8) | (random_value_bytes[2] << 16) | (random_value_bytes[3] << 24));
           
            double random_value = (double)random_value_int / (double)UINT32_MAX;

            if (random_value >= noise_probability) {

                unsigned char noise_type_bytes[1];
                randombytes(noise_type_bytes, sizeof(noise_type_bytes));  

                int noise_type = noise_type_bytes[0] % 2;
                if (noise_type == 0) {
                    count_non_zero_coeffs++;
                } else if (count_non_zero_coeffs > 0) {
                    count_non_zero_coeffs--;
                }
            }

            votes[vote] = count_non_zero_coeffs;
            printf("For vote is %d; random_value is %f; count_non_zero_coeffs is %d\n", vote, random_value, count_non_zero_coeffs);
        }

        int max_count = 0, result = 0;
        for (int i = 0; i < c_base_vote_limit; i++) {
            int count = 0;
            for (int j = 0; j < c_base_vote_limit; j++) {
                if (votes[j] == votes[i]) {
                    count++;
                }
            }
            if (count > max_count) {
                max_count = count;
                result = votes[i];
            }
        }
        count_non_zero_coeffs = result;

        printf("Fianl no_true_collisions is %d; count_non_zero_coeffs is %d\n", no_true_collisions, count_non_zero_coeffs);

        if (no_true_collisions == count_non_zero_coeffs && no_true_collisions == 1) {
            (*no_single_collisions)++;
        #if DO_ATTACK == 0
            got_minus_one = 1;
        #endif
        }
        if (no_true_collisions == count_non_zero_coeffs && no_true_collisions > 1) {
            (*no_multiple_collisions)++;
        }
        if (no_true_collisions != count_non_zero_coeffs && no_true_collisions > count_non_zero_coeffs) {
            (*no_false_negative_collisions)++;
        }
        if (no_true_collisions != count_non_zero_coeffs && no_true_collisions < count_non_zero_coeffs) {
            (*no_false_positive_collisions)++;
            (*false_positive_count)++;
        }
        printf("false_positive_count %d;\n", *false_positive_count);
         
        #if (COLL_CHECK == 1)
        for (int i = 0, index_ones = 0; i < param_p; i++) 
        {
            if (er_decrypt[i] == 1 || er_decrypt[i] == -1) {
                collision_array_index[index_ones] = i;
                collision_array_value[index_ones] = er_decrypt[i];
                index_ones++;
            }
        }
        #endif
        
        count_minus_ones = 0;
        for(int i = 0; i < p; i++)
        {
            if(er_decrypt[i] == -1 || er_decrypt[i] == +1)
            count_minus_ones = count_minus_ones+1;
        }
           
        if(count_minus_ones > 0)
        {
            #if DO_ATTACK == 1
            got_minus_one = 1;
            #endif

        }

        if (got_minus_one == 1) 
        {
            *success_trial = 1;
            printf("Found Non Zero e....\n");

            #if (DO_PRINT == 1)
            FILE *f2 = fopen(ct_file_now_basic, "a");
            if (f2 == NULL) {
                fprintf(stderr, "Error opening file %s for writing\n", ct_file_now_basic);
                exit(1);
            }
            for (int pp1 = 0; pp1 < kem_CIPHERTEXTBYTES; pp1++) {
                fprintf(f2, "%02x", ct[pp1]);
            }
            fclose(f2);
            #endif
        } else 
        {
            #if (DO_PRINT == 1)
            FILE *f2 = fopen(ct_file_basic_failed, "a");
            if (f2 == NULL) {
                fprintf(stderr, "Error opening file %s for writing\n", ct_file_basic_failed);
                exit(1);
            }
            for (int pp1 = 0; pp1 < kem_CIPHERTEXTBYTES; pp1++) {
                fprintf(f2, "%02x", ct[pp1]);
            }
            fclose(f2);
            #endif
        }
    }
}


void get_oracle_response(unsigned char *ct, unsigned char *ss, unsigned char *ss1, unsigned char *pk, unsigned char *sk,
                         int **sec_index_second_part, 
                         int *get_er_decrypt_array_all, int check1_array[], int param_list1_index, 
                         int param_list2_index, int start_idx, int end_idx, int *overall_oracle_count) 
{
    int get_er_decrypt_array[1];
    int count_get_er_decrypt_array_all = 0;
    int array_size = end_idx - start_idx;

    for (int idx1 = start_idx; idx1 < end_idx; idx1++) 
    {
        for (int i = 0; i < u_column_size; i++) 
        {
            sec_index[i] = sec_index_second_part[idx1][i];
        }

        int check1 = check1_array[idx1];
        mul_value = (check1 == 0) ? 1 : -1;

        
        int vote_count_0 = 0;   
        int vote_count_neg1 = 0; 

        crypto_kem_enc(ct, ss, pk);
        crypto_kem_dec(ss1, ct, sk);

        (*overall_oracle_count)++;

        weight_hh = 0;
        for (int jh = 0; jh < p; jh++) 
        {
            if (abs(er_decrypt[jh]) > 0) 
            {
                weight_hh++;
            }
        }

       
        int current_result = (weight_hh != 0) ? -1 : 0;

        // Use randombytes() to generate random numbers to introduce Bernoulli noise
        unsigned char random_value_bytes[4];
        randombytes(random_value_bytes, sizeof(random_value_bytes));  

       
        uint32_t random_value_int = (random_value_bytes[0] | (random_value_bytes[1] << 8) | (random_value_bytes[2] << 16) | (random_value_bytes[3] << 24));
        
        double random_value = (double)random_value_int / (double)UINT32_MAX;

        double noise_probability = 0.96;
        if (random_value >= noise_probability) 
        {
            current_result = (current_result == -1) ? 0 : -1;
        }

        if (current_result == 0) {
            vote_count_0++;
        } else if (current_result == -1) {
            vote_count_neg1++;
        }

        get_er_decrypt_array[0] = (vote_count_0 >= vote_count_neg1) ? 0 : -1;

        if (count_get_er_decrypt_array_all < array_size) {
            get_er_decrypt_array_all[count_get_er_decrypt_array_all] = get_er_decrypt_array[0];
            count_get_er_decrypt_array_all++;
        } else {
            printf("Error: get_er_decrypt_array_all index out of bounds.\n");
        }
    }
}




bool fun_3(int *current_state, int match_table[9][1], 
    int await_value[9][10], int *current_dis, 
    int column_2[4][1], int await_match_table_2[2][3], 
    int *current_tree_state) 
{
bool state_updated = false;

switch (*current_state) 
{
     case 0:

     if (
         match_table[0][0] == await_value[0][4] &&
         match_table[1][0] == await_value[1][4] &&
         match_table[2][0] == await_value[2][4] &&
         match_table[3][0] == await_value[3][4] &&
         match_table[4][0] == await_value[4][4] &&
         match_table[5][0] == await_value[5][4] &&
         match_table[6][0] == await_value[6][4] &&
         match_table[7][0] == await_value[7][4] &&
         match_table[8][0] == await_value[8][4])
     {
         state_updated = true;
         *current_state = 0;
         *current_dis = 1;
         *current_tree_state = 1;
     }
     else if (
         match_table[0][0] == await_value[0][5] &&
         match_table[1][0] == await_value[1][5] &&
         match_table[2][0] == await_value[2][5] &&
         match_table[3][0] == await_value[3][5] &&
         match_table[4][0] == await_value[4][5] &&
         match_table[5][0] == await_value[5][5] &&
         match_table[6][0] == await_value[6][5] &&
         match_table[7][0] == await_value[7][5] &&
         match_table[8][0] == await_value[8][5])
     {
         state_updated = true;
         *current_state = 0;
         *current_dis = 1;
         *current_tree_state = 2;
     }

     else
     {
         state_updated = false;
         *current_state = 0;
         *current_dis = 1;
     }
     break;
 case 1:
     if (column_2[0][0] == await_match_table_2[0][0] && 
         column_2[1][0] == await_match_table_2[1][0]) 
     {
         state_updated = false;  
         *current_state = 1;
         *current_dis = 2;
         printf("Column matches with the first column of await_match_table_2.\n");
     } 
     else if (column_2[0][0] == await_match_table_2[0][1] && 
              column_2[1][0] == await_match_table_2[1][1]) 
     {
         printf("Column matches with the second column of await_match_table_2.\n");
         state_updated = true;           
         *current_state = 1;
         *current_dis = 2;
         *current_tree_state = 5;
     } 
     else if (column_2[0][0] == await_match_table_2[0][2] && 
              column_2[1][0] == await_match_table_2[1][2]) 
     {
         printf("Column matches with the third column of await_match_table_2.\n");
         state_updated = false;  
         *current_state = 1;
         *current_dis = 2;
     }
     break;
 case 2:
     if (column_2[0][0] == await_match_table_2[0][0] && 
         column_2[1][0] == await_match_table_2[1][0]) 
     {
         state_updated = false;  
         *current_state = 2;
         *current_dis = 2;
         printf("Column matches with the first column of await_match_table_2.\n");
     } 
     else if (column_2[0][0] == await_match_table_2[0][1] && 
              column_2[1][0] == await_match_table_2[1][1]) 
     {
         printf("Column matches with the second column of await_match_table_2.\n");
         state_updated = true;             
         *current_state = 2;
         *current_dis = 2;
         *current_tree_state = 6;
     } 
     else if (column_2[0][0] == await_match_table_2[0][2] && 
              column_2[1][0] == await_match_table_2[1][2]) 
     {
         printf("Column matches with the third column of await_match_table_2.\n");
         state_updated = false;  
         *current_state = 2;
         *current_dis = 2;
     }
     break;
}

printf("After fun_3 call: state_updated = %d, current_state = %d, current_dis = %d, current_tree_state = %d\n", 
    state_updated, *current_state, *current_dis, *current_tree_state);
return state_updated;
}


// Node structure for the tree
typedef struct Node {
    bool ge_flag;   
    int value;      
    struct Node *left;  
    struct Node *right; 
} Node;

// SimpleOracle structure
typedef struct {
    double p_prob;      
    int oracle_calls;   
} SimpleOracle;

// False Positive/Negative Oracle structure
typedef struct {
    double **p_positional; 
    int oracle_calls;      
} FalsePositiveNegativePositionalOracle;

// Function to create a SimpleOracle
SimpleOracle* create_simple_oracle(double p_prob) {
    SimpleOracle *oracle = (SimpleOracle*) malloc(sizeof(SimpleOracle));
    oracle->p_prob = p_prob;
    oracle->oracle_calls = 0;
    return oracle;
}

// Create a node for the tree
Node* create_node(bool ge_flag, int value) {
    Node *node = (Node*) malloc(sizeof(Node));
    node->ge_flag = ge_flag;
    node->value = value;
    node->left = NULL;
    node->right = NULL;
    return node;
}

// Recursive function to build a tree from an array of nodes
Node* recursive_tree_from_array(Node *arr[], int i, int n) {
    if (i >= n || arr[i] == NULL) {
        return NULL;
    }

    Node *root = arr[i];
    root->left = recursive_tree_from_array(arr, 2 * i + 1, n);
    root->right = recursive_tree_from_array(arr, 2 * i + 2, n);
    return root;
}

// Predict a bit using SimpleOracle
int simple_oracle_predict_bit(SimpleOracle *oracle, int actual_bit, int pos) {
    oracle->oracle_calls += 1;
    double rnd = (double) rand() / RAND_MAX;
   
    printf("Random number (rnd): %.6f, Oracle accuracy (p_prob): %.6f, Actual bit: %d\n", rnd, oracle->p_prob, actual_bit);
    
    if (rnd < oracle->p_prob) {
        return actual_bit;
    } else {
        return 1 - actual_bit;
    }
}

int* sample_coef_with_simple_oracle(SimpleOracle *oracle, int actual_coef, Node *coding_tree, int *out, int *size) {
    Node *node = coding_tree;
    *size = 0;
    
    while (node != NULL) {
        int expected_bit = (node->ge_flag) ? (actual_coef >= node->value) : (actual_coef <= node->value);
        
        // Set b = -1 to represent the right subtree, and 0 to represent the left subtree
        int b = (expected_bit == 1) ? -1 : 0;
        out[(*size)++] = b;

        if (b == -1) {
            node = node->right;
        } else {
            node = node->left;
        }
    }

    return out;
}

// Function to create a FalsePositiveNegativePositionalOracle
FalsePositiveNegativePositionalOracle* create_fp_fn_oracle(long double **p_positional, int size) 
{
    FalsePositiveNegativePositionalOracle *oracle = (FalsePositiveNegativePositionalOracle *)malloc(sizeof(FalsePositiveNegativePositionalOracle));

    oracle->p_positional = p_positional;  
    oracle->oracle_calls = 0;

    return oracle;
}



// Modified Find value in secret distribution
long double find_in_secret_distrib(long double *Pr, int key, int sum_weight) {
    int index = key + sum_weight;

    if (index >= 0 && index < MAX_SECRET) {
        return Pr[index];  
    } else {
        return 0.0L; 
    }
}

// Calculate probability using FalsePositiveNegativePositionalOracle
long double fp_fn_oracle_prob_of(FalsePositiveNegativePositionalOracle *oracle, int expected, int actual, int pos) {
    oracle->oracle_calls += 1;
    long double *probs = oracle->p_positional[pos];
    long double pr_fp = probs[0], pr_fn = probs[1];
   
    if (expected == 0) {
        return (actual == -1) ? pr_fp : 1.0L - pr_fp;
    } else {
        return (actual == 0) ? pr_fn : 1.0L - pr_fn;
    }
}

long double pr_cond_yx_adaptive(int *y, int y_len, int s, FalsePositiveNegativePositionalOracle *pr_oracle, Node *coding_tree, int *y_prob) {
    long double res = 1.0L;  // Use long double
    Node *node = coding_tree;

    for (int i = 0; i < y_len; i++) {
        if (node == NULL) {
            break;
        }

        int expected_bit = (node->ge_flag) ? (s >= node->value) : (s <= node->value);
        long double prob = (long double)fp_fn_oracle_prob_of(pr_oracle, expected_bit, y[i], y_prob[i]);  // Cast to long double
        res *= prob;

        if (y[i] == -1) {
            node = node->right;
        } else {
            node = node->left;
        }
    }
    return res;
}

long double pr_cond_xy_adaptive(int s, int *y, int y_len, FalsePositiveNegativePositionalOracle *pr_oracle, long double *Pr, int sum_weight, Node *coding_tree, int *y_prob) {
    long double pr_y = 0.0L;  // Use long double

    for (int i = -sum_weight; i <= sum_weight; i++) {
        long double pr_x_y = find_in_secret_distrib(Pr, i, sum_weight) * 
                             pr_cond_yx_adaptive(y, y_len, i, pr_oracle, coding_tree, y_prob);  // Pass y_prob
        pr_y += pr_x_y;
    }

    return pr_cond_yx_adaptive(y, y_len, s, pr_oracle, coding_tree, y_prob) * 
           find_in_secret_distrib(Pr, s, sum_weight) / pr_y;
}


// Initialize oracle accuracy data with 38 positions
long double **init_oracle_accuracy() {
    long double **oracle_accuracy = (long double **)malloc(1 * sizeof(long double *));
   
    oracle_accuracy[0] = (long double *)malloc(2 * sizeof(long double));
    
    oracle_accuracy[0][0] = 0.045000L;  // False positive probability
    oracle_accuracy[0][1] = 0.045000L;  // False negative probability


    return oracle_accuracy;
}



int MAX_ROWS;

int intended_function;


extern int collision_index;
extern int collision_value;

extern int m;
extern int n;

extern int mul_value;

int c3_value_1, c3_value_2, c3_value_3;

extern int *c_value_for_attack_1_1;
extern int *c_value_for_attack_1_2;
extern int *c_value_for_attack_1_3;

extern int pq_counter;

unsigned char entropy_input[48];
unsigned char seed[NO_TESTS][48];

int check_count;

int zeros;

int c_base_vote_limit; 
int cond_prob_loop;
int state_oracle_now;
int current_state;


int theoretical_loop_count;



int main()
{
    FILE                *fp_req, *fp_rsp;
    int                 ret_val;
    int i;
    unsigned char *ct = 0;
    unsigned char *ss = 0;
    unsigned char *ss1 = 0;
    unsigned char *pk = 0;
    unsigned char *sk = 0;
 
  
    int param_p = 653; 
  

    for (int i = 0; i < 48; i++) 
    {
        entropy_input[i] = get_random() & 0xFF;
        
    }

    randombytes_init(entropy_input, NULL, 256);

    // Generate seed
    for (int i = 0; i < NO_TESTS; i++) {
        randombytes(seed[i], 48); 
    }

    printf("Firstly seed is\n");
    for (int i = 0; i < NO_TESTS; i++) {
        printf("Seed[%d]: ", i);
        for (int j = 0; j < 48; j++) {
            printf("%02x", seed[i][j]);
        }
        printf("\n");
    }



    #if (DO_ATTACK_COLLISION_NEW == 1)

    int found_c = 0;

    FILE * f2;
    FILE * f3;

    #if (DO_PRINT == 1)

    char ct_file_now[30];
    char ct_file_now_basic[30];
    char keypair_file[30];
    char oracle_responses_now_file_name[30];
    char ct_file_basic_failed[50];

    // We can store the data of a single iteration in files...
    // Please note that these files will be overwritten for every iteration...
    // We store the oracle responses in oracle_resp.bin...

    // Here, we store the attack ciphertexts...
    sprintf(oracle_responses_now_file_name,"oracle_resp.bin");

    // Here, we store the attack ciphertexts...
    sprintf(ct_file_now,"ct_file_now.bin");

    // Here, we store the base ciphertext...
    sprintf(ct_file_now_basic,"ct_file_basic.bin");

    // Here, we store the public and private key pair...
    sprintf(keypair_file,"keypair_file.bin");

    // Here, we store the failed ciphertexts which do not correspond to any collision...
    sprintf(ct_file_basic_failed,"ct_file_basic_failed.bin");

    // Clear the contents of the d1_d2_data.txt file
    FILE *d1_d2_data_file_txt = fopen("d1_d2_data.txt", "w");
    if (d1_d2_data_file_txt == NULL) {
        perror("Failed to open file");
        return 1;
    }
    fclose(d1_d2_data_file_txt);
    
    // Open and clear the file to store the private key and collision information
    FILE *private_key_file = fopen("private_key_and_collision_info.bin", "w");
    if (private_key_file == NULL) {
        printf("Failed to open file: private_key_and_collision_info.bin\n");
        return 1;
    }

    // Open a new file to store actual_match_table
    FILE *actual_match_table_file = fopen("actual_match_table.bin", "w");
    if (actual_match_table_file == NULL) {
        printf("Failed to open file: actual_match_table.bin\n");
        return 1;
    }

    // Open and clear the file for all test actual and theoretical oracle responses
    FILE *after_change_ldpc_output_file = fopen("after_change_ldpc.bin", "w");
    if (after_change_ldpc_output_file == NULL) {
        printf("Failed to open file: ldpc.bin\n");
        return 1;
    }
    fclose(after_change_ldpc_output_file);

    // Clear alpha_u_and_conditional_probabilities_of_%d.bin files
    for (int coll_index = p-1; coll_index >= 0; coll_index--) 
    {
      char filename[256];
      sprintf(filename, "When 1 %d for alpha_u_and_conditional_probabilities.bin", coll_index);
      FILE *file1 = fopen(filename, "w");
      if (file1 == NULL) {
          printf("Failed to open file: %s\n", filename);
          return 1;
      }
      fclose(file1);

    }
 

    // Open and clear the file "beta_u_and_theoretical_oracle_response"
    FILE *beta_u_and_theoretical_oracle_response_file = fopen("beta_u_and_theoretical_oracle_response.bin", "w");
    if (beta_u_and_theoretical_oracle_response_file == NULL) {
        printf("Failed to open file: beta_u_and_theoretical_oracle_response.bin\n");
        return 1;
    }
    fclose(beta_u_and_theoretical_oracle_response_file);  

    // 打开并清空文件 "actual_oracle_response_for_param_list"
    FILE *actual_oracle_response_for_param_list_file = fopen("actual_oracle_response_for_param_list.bin", "w");
    if (actual_oracle_response_for_param_list_file == NULL) {
        printf("Failed to open file: actual_oracle_response_for_param_list.bin\n");
        return 1;
    }
    fclose(actual_oracle_response_for_param_list_file);
    
    // Open and clear the file "actual_oracle_response_for_param_list"
    FILE *actual_oracle_response_file = fopen("actual_oracle_response.txt", "w");
    if (actual_oracle_response_file == NULL)
    {
        printf("Failed to open file: actual_oracle_response.txt\n");
        return 1;
    }
    fclose(actual_oracle_response_file);
 
    // Open and clear the file "u_1_u_2_logs.txt"
    FILE *u_1_u_2_logs_file = fopen("u_1_u_2_logs.txt", "w");
    if (u_1_u_2_logs_file == NULL)
    {
        printf("Failed to open file: u_1_u_2_logs.txt\n");
        return 1;
    }
    fclose(u_1_u_2_logs_file);  

    FILE *secret_key_weight_file = fopen("secret_key_weight.txt", "w");
    if (secret_key_weight_file == NULL)
    {
        printf("Failed to open file: secret_key_weight.txt\n");
        return 1;
    }
    fclose(secret_key_weight_file);  

    FILE *ldpc_final_output_file = fopen("ldpc_final_output.txt", "w");
    if (ldpc_final_output_file == NULL)
    {
        printf("Failed to open file: ldpc_final_output.txt\n");
        return 1;
    }
    fclose(ldpc_final_output_file);  

    FILE *overall_oracle_calls_count_file = fopen("Overall_oracle_calls_count.txt", "w");
    if (overall_oracle_calls_count_file == NULL)
    {
        printf("Failed to open file: Overall_oracle_calls_count.txt\n");
        return 1;
    }
    fclose(overall_oracle_calls_count_file);

    FILE *ldpc_fprime_output_file = fopen("ldpc_fprime_output.txt", "w");
    if (ldpc_final_output_file == NULL)
    {
        printf("Failed to open file: ldpc_fprime_output.txt\n");
        return 1;
    }
    fclose(ldpc_fprime_output_file); 

    
    #endif
   

    
    m = M_VALUE;
    n = N_VALUE;

    // This is used to calculate k1 and k2 for the base ciphertext cbase, as described in the paper...

    int max_distance = 1000000;
    int max_distance2 = 0;
    int max_min_distance = 0;
    while(found_c == 0)
    {
      for(int hg = 0; hg < q; hg++)
      {

        for(int hg1 = 0; hg1 < q; hg1++)
        {
              int value1 = hg * (3*2*m) + hg1 * (2*n);

              int touch_np = 0;

              max_distance = 1000000;

              for(int poss = 0; poss <= 2*m; poss++)
              {
                for(int poss1 = 0; poss1 <= 2*n; poss1++)
                {
                      if(!(poss == 2*m && poss1 == 2*n))
                      {
                        int value2 = hg * (3*poss) + hg1*poss1;

                        if(max_distance > (abs(q12 - value2)))
                          max_distance = (abs(q12 - value2));

                        if((value1 < q12) || (value2 > q12)
                        || (hg%3 != 0) || (hg1%3 != 0) || (abs(q12 - value1) < C_VALUE_THRESHOLD_1) || (abs(q12 - value2) < C_VALUE_THRESHOLD_2))
                        {
                          touch_np = 1;
                        }

                      }
                }
              }
              int local_max_min_distance = min(max_distance, abs(q12 - value1));

              if(touch_np == 0)
              {
                found_c = 1;

                if(local_max_min_distance > max_min_distance)
                {
                  c_value_1 = hg;
                  c_value_2 = hg1;
                  max_distance2 = max_distance;
                  max_min_distance = local_max_min_distance;
                  printf("hg = %d, hg1 = %d, Diff1: %d, Diff2: %d\n", hg, hg1, abs(q12 - value1), max_distance2);
                }
              }
        }
      }
    }



    int list_of_c1_values[30][10];
    int list_of_c2_values[30][10];


    int no_c1_values = 0;
    int no_c2_values = 0;

    int sample_1, sample_2, sample_3;

    printf("Found c and q is %d\n", q);
    
  
    // Here, we are trying to compute l1, l2 and l3 for the attack ciphertexts as shown in the paper...
    // We compute l1, l2, l3 is used to distinguish 1...

    max_distance = 1000000;
    max_distance2 = 0;
    max_min_distance = 0;
    int limit_value = 1000;
    
    float max_weight_score = 0;

    int bound = 3; //
    c_value_for_attack_1_1 = malloc(bound * sizeof(int));
    c_value_for_attack_1_2 = malloc(bound * sizeof(int));
    c_value_for_attack_1_3 = malloc(bound * sizeof(int));
    
    for (int dist_value = 1; dist_value < bound +1; dist_value++)
    {
      max_distance = 1000000;
      max_distance2 = 0;
      max_min_distance = 0;
      printf("distinguishing value %d\n", dist_value);
      for(int hg = 0; hg < limit_value; hg = hg+3)
      {
        for(int hg1 = 0; hg1 < limit_value; hg1 = hg1+3)
        {
          for(int hg2 = 0; hg2 < limit_value; hg2 = hg2+3)
          {
            sample_1 = hg;
            sample_2 = hg1;
            sample_3 = hg2;

            int value1 = sample_1 * (3*2*m) + sample_2 * (2*n) + sample_3 * (3*dist_value);

            if (abs(q12 - value1) < max_min_distance) {
              continue;
            }

            int value2;

            int touch_np = 0;

            max_distance = 1000000;


            for(int poss = 0; poss <= 2*m; poss++)
            {
              for(int poss1 = 0; poss1 <= 2*n; poss1++)
              {
                for(int poss2 = 0; poss2 <= bound; poss2++) //bound
                {

                  if(!((poss == 2*m) && (poss1 == 2*n) && (poss2 >= dist_value)))
                  {

                    value2 = sample_1 * (3*poss) + sample_2 * poss1 + sample_3 * (3*poss2);
 
                    if(max_distance > (abs(q12 - value2)))
                      max_distance = (abs(q12 - value2));

                    if((value1 < q12) || (value2 > q12))
                      // || (abs(value1 - q12) < GAP_THRESHOLD_1_1) || (abs(value2 - q12) < GAP_THRESHOLD_1_2))
                    {
                      touch_np = 1;
                    }
                  }

                }
              }
            }
            if(max_distance >= 35)
            {
              float weight_score = 1.0 *abs(q12 - value1) + 1.0 * max_distance;

                if(touch_np == 0)
                {
                    
                    if (dist_value == 1 && weight_score >= 0 && abs(q12 - value1) >= max_distance)
                    {
                        c_value_for_attack_1_1[dist_value - 1] = hg;
                        c_value_for_attack_1_2[dist_value - 1] = hg1;
                        c_value_for_attack_1_3[dist_value - 1] = hg2;
                        max_distance2 = max_distance;
                        max_weight_score = weight_score;
                        printf("hg = %d, hg1 = %d, hg2 = %d, Diff1: %d, Diff2: %d, weight_score: %f\n", hg, hg1, hg2, abs(q12 - value1), max_distance2, weight_score);
                    }
                    else if (dist_value == 2 && weight_score >= 100 && abs(q12 - value1) >= max_distance)
                    {
                        c_value_for_attack_1_1[dist_value - 1] = hg;
                        c_value_for_attack_1_2[dist_value - 1] = hg1;
                        c_value_for_attack_1_3[dist_value - 1] = hg2;
                        max_distance2 = max_distance;
                        max_weight_score = weight_score;
                        printf("hg = %d, hg1 = %d, hg2 = %d, Diff1: %d, Diff2: %d, weight_score: %f\n", hg, hg1, hg2, abs(q12 - value1), max_distance2, weight_score);
                    }
                    else if (dist_value == 3 && weight_score >= 200 && abs(q12 - value1) >= max_distance)
                    {
                        c_value_for_attack_1_1[dist_value - 1] = hg;
                        c_value_for_attack_1_2[dist_value - 1] = hg1;
                        c_value_for_attack_1_3[dist_value - 1] = hg2;
                        max_distance2 = max_distance;
                        max_weight_score = weight_score;
                        printf("hg = %d, hg1 = %d, hg2 = %d, Diff1: %d, Diff2: %d, weight_score: %f\n", hg, hg1, hg2, abs(q12 - value1), max_distance2, weight_score);
                    }
                }
            }             
          }
        }
      }
    }
     

    double profile_average_count = 0;
    double trace_average_count = 0;

    // Iterate over the number of tests you want to run... The NO_TESTS variable is defined in params.h header file...

    if (!ct) ct = malloc(crypto_kem_CIPHERTEXTBYTES);
    if (!ct) abort();
    if (!ss) ss = malloc(crypto_kem_BYTES);
    if (!ss) abort();
    if (!ss1) ss1 = malloc(crypto_kem_BYTES);
    if (!ss1) abort();
    if (!pk) pk = malloc(crypto_kem_PUBLICKEYBYTES);
    if (!pk) abort();
    if (!sk) sk = malloc(crypto_kem_SECRETKEYBYTES);
    if (!sk) abort();

    int multiple_collision_count = 0;
   
    int count_non_zero_coeffs = 0;
    int total_profile_trials_overall = 0;

    int no_single_collisions = 0;
    int no_multiple_collisions = 0;
    int no_false_negative_collisions = 0;
    int no_false_positive_collisions = 0;
    
    for (int pq=0; pq<NO_TESTS; pq++)
    {
        pq_counter = pq;  
        
        #if (DO_PRINT == 1)

        f2 = fopen(oracle_responses_now_file_name, "w+");
        fclose(f2);

        f2 = fopen(ct_file_basic_failed, "w+");
        fclose(f2);

        f2 = fopen(keypair_file, "w+");
        fclose(f2);

        f2 = fopen(ct_file_now_basic, "w+");
        fclose(f2);

        f2 = fopen(ct_file_now, "w+");
        fclose(f2);

        #endif

        printf("Trial: %d\n",pq);
        
        printf("Using seed[%d]: ", pq);
        for (int j = 0; j < 48; j++) {
            printf("%02x", seed[pq][j]);
        }
        printf("\n");
        randombytes_init(seed[pq], NULL, 256);

        printf("***********Testing for New Key***********\n");
        if ( (ret_val = crypto_kem_keypair(pk, sk)) != 0)
        {
            return KAT_CRYPTO_FAILURE;
        }

        #if (DO_PRINT == 1)

        f2 = fopen(keypair_file, "a");

        for(int pp1=0;pp1<crypto_kem_PUBLICKEYBYTES;pp1++)
        {
          fprintf(f2,"%02x", pk[pp1]);
        }

        for(int pp1=0;pp1<crypto_kem_SECRETKEYBYTES;pp1++)
        {
          fprintf(f2,"%02x", sk[pp1]);
        }
        fclose(f2);

        #endif


        printf("Count of Single Collisions: %d\n", no_single_collisions);
        printf("Count of Multiple Collisions: %d\n", no_multiple_collisions);
        printf("Count of False Negative Collisions: %d\n", no_false_negative_collisions);
        printf("Count of False Positive Collisions: %d\n", no_false_positive_collisions);
        printf("Count of Trials Overall: %d\n", total_profile_trials_overall);

        printf("Probability of Single Collisions: %f\n", ((float)no_single_collisions/total_profile_trials_overall));
        printf("Probability of Multiple Collisions: %f\n", ((float)no_multiple_collisions/total_profile_trials_overall));
        printf("Probability of False Negative Collisions: %f\n", ((float)no_false_negative_collisions/total_profile_trials_overall));
        printf("Probability of False Positive Collisions: %f\n", ((float)no_false_positive_collisions/total_profile_trials_overall));

        int successful_attack_done = 0;
        int overall_oracle_count = 0;
    

    //    while (successful_attack_done == 0)
    //    {
                // Step 1
                rej:
                printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%TRIAL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
                      
                       
                collision_count = 0;  

                int collision_array_index[param_p]; 
                int collision_array_value[param_p]; 
                int false_positive_count = 0;  
    
                for (int i = 0; i < param_p; i++) {
                    collision_array_index[i] = 0;
                    collision_array_value[i] = 0;
                }

                intended_function = 0;
                c_base_vote_limit = 9;
                
                int success_trial = 0;
                generate_base_ciphertext(&success_trial, ct, ss, pk, sk, param_p, &no_single_collisions, &no_multiple_collisions, &no_false_negative_collisions, &no_false_positive_collisions, crypto_kem_CIPHERTEXTBYTES, ct_file_now_basic, ct_file_basic_failed, collision_array_index, collision_array_value, &collision_count,&total_profile_trials_overall, &false_positive_count, c_base_vote_limit, &overall_oracle_count);
             
                #if (DO_ATTACK == 1)

                #if (COLL_CHECK == 1)
                
                printf("Collisions detected: %d\n", collision_count);
                for (int i = 0; i < collision_count; i++) {
                    printf("collision_index[%d]: %d, collision_value[%d]: %d\n", i, collision_array_index[i], i, collision_array_value[i]);
                }
                #endif
                
                #if (COLL_CHECK == 1)
                
                FILE *secret_key = fopen("private_key_and_collision_info.bin", "a");
                if (secret_key == NULL) {
                    printf("Failed to open file: private_key_and_collision_info.bin\n");
                    return 1;
                }

                fprintf(secret_key, "pq_counter: %d\n", pq);
                if (collision_count > 0) {
                    for (int i = 0; i < collision_count; i++) {
                        fprintf(secret_key, "collision_index[%d]: %d, collision_value[%d]: %d\n", i, collision_array_index[i], i, collision_array_value[i]);
                        if (false_positive_count > 0) {
                            fprintf(secret_key, "false positive happened\n");
                        }
                    }
                } else {
                    fprintf(secret_key, "false positive happened\n");
                }
                fprintf(secret_key, "The private key is:\n");
                for (int i = 0; i < param_p; i++) {
                    fprintf(secret_key, "%d,", global_f[i]);
                }
                fprintf(secret_key, "\n");

                fclose(secret_key);
                #endif


                int tree_size_1 = 0;
                int initial_capacity = 50; 
                Node **coding_arr_1 = (Node **)malloc(initial_capacity * sizeof(Node *));

                if (coding_arr_1 == NULL) {
                    printf("Memory allocation failed for first tree\n");
                    return 0;
                }

                collision_index = collision_array_index[0];
                collision_value = collision_array_value[0];
               
                int oracle_response_2_capacity = 3 * p;  
                int **oracle_response_2 = NULL;  
                int oracle_response_2_size = 0;  
                int *oracle_response_2_col_sizes = NULL;  

                if (oracle_response_2 != NULL) {
                    for (int i = 0; i < oracle_response_2_capacity; i++) {
                        free(oracle_response_2[i]); 
                    }
                    free(oracle_response_2); 
                    oracle_response_2 = NULL; 
                }

                if (oracle_response_2_col_sizes != NULL) {
                    free(oracle_response_2_col_sizes); 
                    oracle_response_2_col_sizes = NULL; 
                }

                oracle_response_2 = (int **)malloc(oracle_response_2_capacity * sizeof(int *));
                oracle_response_2_col_sizes = (int *)malloc(oracle_response_2_capacity * sizeof(int));  

                if (oracle_response_2 == NULL || oracle_response_2_col_sizes == NULL) {
                    printf("Memory allocation failed for oracle_response_2 or oracle_response_2_col_sizes\n");
                    return 1;
                }

                for (int i = 0; i < oracle_response_2_capacity; i++) {
                    oracle_response_2[i] = NULL;  
                    oracle_response_2_col_sizes[i] = 0;  
                }   


                int Pos_neg_index_second_capacity = 3 * p; 
                int **Pos_neg_index_second = NULL; 
                int Pos_neg_index_second_size = 0;
                int *Pos_neg_index_second_col_sizes = NULL; 

                if (Pos_neg_index_second != NULL) {
                    for (int i = 0; i < Pos_neg_index_second_capacity; i++) {
                        free(Pos_neg_index_second[i]); 
                    }
                    free(Pos_neg_index_second); 
                    Pos_neg_index_second = NULL;
                }
                if (Pos_neg_index_second_col_sizes != NULL) {
                    free(Pos_neg_index_second_col_sizes); 
                    Pos_neg_index_second_col_sizes = NULL;
                }                

                
                Pos_neg_index_second = (int **)malloc(Pos_neg_index_second_capacity * sizeof(int *));
                Pos_neg_index_second_col_sizes = (int *)malloc(Pos_neg_index_second_capacity * sizeof(int)); 

                if (Pos_neg_index_second == NULL || Pos_neg_index_second_col_sizes == NULL) {
                    printf("Memory allocation failed for Pos_neg_index_second or Pos_neg_index_second_col_sizes\n");
                    return 1;
                }

               
                for (int i = 0; i < Pos_neg_index_second_capacity; i++) {
                    Pos_neg_index_second[i] = NULL;  
                    Pos_neg_index_second_col_sizes[i] = 0; 
                }


                ParamRecord *param_records = NULL;  
                int param_record_count = 0;         
                int param_record_capacity = 20;     

                param_records = (ParamRecord *)malloc(param_record_capacity * sizeof(ParamRecord));
                if (param_records == NULL) {
                    printf("Memory allocation failed for param_records\n");
                    return 1;
                }


                int *check1_array = NULL;  

                int **filtered_sec_index_second_part = NULL;   
                int *filtered_check2_array = NULL;            
                int *filtered_position_second_array = NULL;   
                int filtered_array_second_length = 0;         
                int filtered_position_second_array_length = 0;


                int **updated_sec_index_second_part = NULL;
                int *updated_check1_array_2 = NULL;
                int *updated_index_array_2 = NULL;
                int updated_array_size_2 = 0;
                int updated_index_array_size_2 = 0;
 
               
                int **new_sec_index_second_part = NULL;   
                int *new_check1_array_2 = NULL;          
                int *index_array_2 = NULL;          
                int new_array_size_2 = 0;               
                int index_array_size_2 = 0;              
 
                int *get_er_decrypt_array_all = NULL;
                               
                int start_idx;
                int end_idx;
                int check1 = 0;
                int array_size;
              

                current_state = 0;
                int current_dis = 0;
                int current_tree_state = 0;               
                               
                intended_function = 1;
                param_list1_index = 0;  
                param_list2_index = 0;   
                param_which_list=0;
                    
             
                int param_value_to_fill = -1;
                cond_prob_loop =0;                
                                           
                bool state_updated = false;  
                
                while (cond_prob_loop < 1)
                {
                   
                    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%cond_prob_TRIAL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
                    state_oracle_now = 0;                  
                    current_state = 0;  
                    tree_size_1=0;                                          
                    state_updated = false;                     
                    bool use_new_sec_index = false;  
                 
                    while (true)
                    {  
                        if (use_new_sec_index) 
                        {
                            for (int i = 0; i < new_array_size_2; i++) 
                            {
                                sec_index_second_part[i][0] = new_sec_index_second_part[i][0];  
                                sec_index_second_part[i][1] = new_sec_index_second_part[i][1];  
                            }

                            printf("Copied sec_index_second_part contents from new_sec_index_second_part:\n");
                            for (int i = 0; i < new_array_size_2; i++) {
                                printf("sec_index_second_part[%d][0] = %d, sec_index_second_part[%d][1] = %d\n", 
                                    i, sec_index_second_part[i][0], i, sec_index_second_part[i][1]);
                            }
                           
                            if (new_check1_array_2 != NULL) {
                                int *temp_new_check1_array_2_copy = (int *)malloc(new_array_size_2 * sizeof(int));
                                if (temp_new_check1_array_2_copy == NULL) {
                                    printf("Memory allocation failed for temp_new_check1_array_2_copy.\n");
                                    return 1;
                                }

                                memcpy(temp_new_check1_array_2_copy, new_check1_array_2, new_array_size_2 * sizeof(int));
                                check1_array = temp_new_check1_array_2_copy;

                            } else {
                               
                                printf("new_check1_array_2 is NULL, using backup data.\n");
                            }
                            start_idx = 0;
                            end_idx = weight_list_k1;
                            array_size = end_idx - start_idx;  
                           
                            free(get_er_decrypt_array_all);
                            get_er_decrypt_array_all = (int *)malloc(array_size * sizeof(int));
                            if (get_er_decrypt_array_all == NULL) {
                                printf("Memory allocation failed array_sizeis %d\n",array_size);
                                return 1;
                            }
                            
                            get_oracle_response(ct, ss, ss1, pk, sk,
                                                sec_index_second_part, 
                                                get_er_decrypt_array_all, check1_array, 
                                                param_list1_index, param_list2_index, 
                                                start_idx, end_idx, 
                                                &overall_oracle_count);
                            printf("For new sec_index, actual oracle response is...\n");
                            for (int i = 0; i < array_size; i++) {
                                printf("%d,", get_er_decrypt_array_all[i]);
                            }
                            printf("\n");
                            double counts[1][2];                 
                            calculate_counts(get_er_decrypt_array_all, array_size, counts);
                           
                            for (int i = 0; i < 1; i++) {
                                printf("counts[%d][0]: %f,", i, counts[i][0]);
                                printf("counts[%d][1]: %f,", i, counts[i][1]);
                            }
                            printf("\n"); 
                            update_match_table_2(counts, weight_list_k1);
                        
                            printf("current_state is %d\n", current_state);
                            // Call the fun_3 function to determine which column matches
                            state_updated = fun_3(&current_state, column, await_match_table, &current_dis, column_2, await_match_table_2, &current_tree_state);                           
     
                        } 
                        else if (cond_prob_loop == 0) 
                        {
                           
                            // Now compute the actual oracle response for the first 200 pairs of (u1, u2) given (l_11, l_12, l_13); 
                            MAX_ROWS = 3 * p;
                            u_1_u_2 = MAX_ROWS;
                            int num_rows = MAX_ROWS;           
                            int num_cols = 2;             
                            
                            char command[512];

                            sec_index_second_part = (int **)malloc(MAX_ROWS * sizeof(int *));
                            for (int i = 0; i < MAX_ROWS; i++) {
                                sec_index_second_part[i] = (int *)malloc(NUM_COLS * sizeof(int));
                            }

                        
                            // Select the appropriate path, call the external function, and compute 3*p index list values (u1, u2)
                            snprintf(command, sizeof(command),
                                     "/home/fei/C/SCA_Assisted_CCA_on_NTRU/NTRU_Prime/PC_Oracle_based_SCA/Attack_Simulations/sntrup761/SCA-LDPC/python-virtualenv/bin/python3 "
                                     "/home/fei/C/SCA_Assisted_CCA_on_NTRU/NTRU_Prime/PC_Oracle_based_SCA/Attack_Simulations/sntrup761/SCA-LDPC/simulate-with-python/gen_ldpc_matrix.py "
                                     "%d %d %d " OUTPUT_MATRIX_PATH,
                                     p, num_rows, num_cols);
                            int ret = system(command);
                            if (ret != 0) {
                                printf("Error: Python script execution failed!\n");
                                return 1;
                            }
                        
                            // Read output_matrix.txt and store the obtained data in sec_index_second_part
                            load_ldpc_from_file(sec_index_second_part, &u_1_u_2, MAX_ROWS);
                                                      
                            for (int i = 0; i < u_1_u_2; i++) {
                                printf("{%d, %d}\n", sec_index_second_part[i][0], sec_index_second_part[i][1]);
                            }
                            printf("\n");

                           
                            backup_sec_index_second_part = (int **)malloc(MAX_ROWS * sizeof(int *));
                            for (int i = 0; i < MAX_ROWS; i++) {
                                backup_sec_index_second_part[i] = (int *)malloc(num_cols * sizeof(int));
                            }

                            for (int i = 0; i < MAX_ROWS; i++) {
                                memcpy(backup_sec_index_second_part[i], sec_index_second_part[i], num_cols * sizeof(int));
                            }
                            printf("Backup sec_index_second_part contents:\n");
                            for (int i = 0; i < u_1_u_2; i++) {
                                printf("{%d, %d}\n", backup_sec_index_second_part[i][0], backup_sec_index_second_part[i][1]);
                            }
                            printf("\n");
                            
                            u_column_size = 2;  

                            sec_index = (int *)malloc(u_column_size * sizeof(int));
                            if (sec_index == NULL) {
                                printf("Memory allocation failed for sec_index after resizing.\n");
                                return 1;
                            }
                           
                            start_idx = 0;
                            end_idx = weight_list_k1;
                            array_size = end_idx - start_idx;
                           
                            if (get_er_decrypt_array_all != NULL) {
                                free(get_er_decrypt_array_all);
                            }
                            
                            get_er_decrypt_array_all = (int *)malloc(array_size * sizeof(int));
                            if (get_er_decrypt_array_all == NULL) {
                                printf("Memory allocation failed\n");
                                return 1;
                            }

                            if (check1_array != NULL) {
                                free(check1_array);
                            }

                            check1_array = (int *)malloc(array_size * sizeof(int));  
                            if (check1_array == NULL) {
                                printf("Memory allocation failed for check1_array\n");
                                free(get_er_decrypt_array_all); 
                                return 1;
                            }

                            for (int i = 0; i < array_size; i++) {
                                check1_array[i] = 0;
                            }

                            // Call the get_oracle_response function to collect the output of the PC oracle
                            get_oracle_response(ct, ss, ss1, pk, sk,
                                                sec_index_second_part, 
                                                get_er_decrypt_array_all, check1_array, 
                                                param_list1_index, param_list2_index, 
                                                start_idx, end_idx, 
                                                &overall_oracle_count);
                            printf("her1 Actual oracle response is...\n");
                            for (int i = 0; i < array_size; i++) {
                                printf("%d,", get_er_decrypt_array_all[i]);
                            }
                            printf("\n");
                 
                            // According to actual oracle response, compute column...
                            int size = array_size;
                            double counts[1][2];
                            
                            calculate_counts(get_er_decrypt_array_all, size, counts);
                            for (int i = 0; i < 1; i++) {
                                printf("counts[%d][0]: %f,", i, counts[i][0]);
                                printf("counts[%d][1]: %f,", i, counts[i][1]);
                            }
                            printf("\n"); 
    
                            update_match_table(counts, average_match_table);
                                                    
                            // Call the fun_3 function to determine which column matches
                            state_updated = fun_3(&current_state, column, await_match_table, &current_dis, column_2, await_match_table_2, &current_tree_state);                     
                                    
                        }
                      
                        printf("State updated: %d\n", state_updated);
                        printf("Current state: %d\n", current_state);
                        printf("Current dis: %d\n", current_dis);  
                       
                        // Use state_updated to determine whether the expected distinguisher has been obtained
                        if (state_updated) 
                        {  
                            // Record the parameters of the attack ciphertext used                
                            if (param_record_count >= param_record_capacity) {
                                param_record_capacity += 20; 
                                param_records = (ParamRecord *)realloc(param_records, param_record_capacity * sizeof(ParamRecord));
                                if (param_records == NULL) {
                                    printf("Memory reallocation failed for param_records\n");
                                    return 1;
                                }
                            }

                            param_records[param_record_count].param_which_list = param_which_list;
                            param_records[param_record_count].param_list1_index = param_list1_index;
                            param_records[param_record_count].param_list2_index = param_list2_index;
                            param_record_count++; 
                                                      
                            if (current_dis == 1)
                            {                                   
                                param_value_to_fill = 0;  
                                
                                for (int idx2 = 0; idx2 < array_size; idx2++) 
                                {
                                    if (oracle_response_2[idx2] == NULL) {
                                        
                                        oracle_response_2[idx2] = (int *)malloc(1 * sizeof(int));
                                        oracle_response_2_col_sizes[idx2] = 1;
                                        if (oracle_response_2[idx2] == NULL) {
                                            printf("Memory allocation failed for oracle_response_2[%d]\n", idx2);
                                            return 1;
                                        }
                                    } else {
                                       
                                        int current_col_size = oracle_response_2_col_sizes[idx2];
                                        oracle_response_2[idx2] = (int *)realloc(oracle_response_2[idx2], (current_col_size + 1) * sizeof(int));
                                        if (oracle_response_2[idx2] == NULL) {
                                            printf("Memory reallocation failed for oracle_response_2[%d]\n", idx2);
                                            return 1;
                                        }
                                        oracle_response_2_col_sizes[idx2] = current_col_size + 1;
                                    }

                                    int current_col_size = oracle_response_2_col_sizes[idx2];
                                    oracle_response_2[idx2][current_col_size - 1] = get_er_decrypt_array_all[idx2];

                                    if (Pos_neg_index_second[idx2] == NULL) {
                                        Pos_neg_index_second[idx2] = (int *)malloc(1 * sizeof(int));
                                        Pos_neg_index_second_col_sizes[idx2] = 1;
                                        if (Pos_neg_index_second[idx2] == NULL) {
                                            printf("Memory allocation failed for Pos_neg_index_second[%d]\n", idx2);
                                            return 1;
                                        }
                                    } else {
                                        int current_col_size = Pos_neg_index_second_col_sizes[idx2];
                                        Pos_neg_index_second[idx2] = (int *)realloc(Pos_neg_index_second[idx2], (current_col_size + 1) * sizeof(int));
                                        if (Pos_neg_index_second[idx2] == NULL) {
                                            printf("Memory reallocation failed for Pos_neg_index_second[%d]\n", idx2);
                                            return 1;
                                        }
                                        Pos_neg_index_second_col_sizes[idx2] = current_col_size + 1;
                                    }

                                    current_col_size = Pos_neg_index_second_col_sizes[idx2];
                                    Pos_neg_index_second[idx2][current_col_size - 1] = param_value_to_fill;
                                }

                                printf("Contents of oracle_response_2 (Column-wise):\n");
                                for (int col = 0; col < oracle_response_2_capacity; col++) {
                                    printf("Column %d: ", col);
                                    for (int row = 0; row < oracle_response_2_col_sizes[col]; row++) {
                                        printf("%d ", oracle_response_2[col][row]);
                                    }
                                    printf("\n");
                                }

                                printf("Contents of Pos_neg_index_second (Column-wise):\n");
                                for (int col = 0; col < Pos_neg_index_second_capacity; col++) {
                                    printf("Column %d: ", col);
                                    for (int row = 0; row < Pos_neg_index_second_col_sizes[col]; row++) {
                                        printf("%d ", Pos_neg_index_second[col][row]);
                                    }
                                    printf("\n");
                                }

                                Pos_neg_index_second_size++;
                                oracle_response_2_size++;

                                // Now compute the actual oracle response for (l_11, l_12, l_13) using the next 100 pairs of (u1, u2) by querying the PC oracle
                                start_idx = weight_list_k1;
                                end_idx = weight_list_k2;
                                array_size = end_idx - start_idx;
                               
                                free(get_er_decrypt_array_all);
                                get_er_decrypt_array_all = (int *)malloc(array_size * sizeof(int));
                                if (get_er_decrypt_array_all == NULL) {
                                    printf("Memory allocation failed\n");
                                    return 1;
                                }
                               
                                free(check1_array);  
                                check1_array = (int *)malloc(weight_list_k2 * sizeof(int)); 
                                if (check1_array == NULL) {
                                    printf("Memory allocation failed for check1_array\n");
                                    return 1;
                                }

                                for (int i = 0; i < weight_list_k2; i++) {
                                    check1_array[i] = 0;
                                }
                               
                                get_oracle_response(ct, ss, ss1, pk, sk,
                                                    sec_index_second_part, 
                                                    get_er_decrypt_array_all, check1_array, 
                                                    param_list1_index, param_list2_index, 
                                                    start_idx, end_idx, 
                                                    &overall_oracle_count);
                               
                                printf("For new sec_index, actual oracle response is...\n");
                                for (int i = 0; i < array_size; i++) {
                                    printf("%d,", get_er_decrypt_array_all[i]);
                                }
                                printf("\n"); 

                                for (int idx1 = start_idx; idx1 < end_idx; idx1++) 
                                {
                                    // --- Store oralce response into oracle_response_2 ---
                                    if (oracle_response_2[idx1] == NULL) {
                                        oracle_response_2[idx1] = (int *)malloc(1 * sizeof(int));
                                        oracle_response_2_col_sizes[idx1] = 1;
                                        if (oracle_response_2[idx1] == NULL) {
                                            printf("Memory allocation failed for oracle_response_2[%d]\n", idx1);
                                            return 1;
                                        }
                                    } else {
                                        int current_col_size = oracle_response_2_col_sizes[idx1];
                                        oracle_response_2[idx1] = (int *)realloc(oracle_response_2[idx1], current_col_size * sizeof(int));
                                        if (oracle_response_2[idx1] == NULL) {
                                            printf("Memory reallocation failed for oracle_response_2[%d]\n", idx1);
                                            return 1;
                                        }
                                    }

                                    int current_col_size = oracle_response_2_col_sizes[idx1];
                                    oracle_response_2[idx1][current_col_size - 1] = get_er_decrypt_array_all[idx1 - start_idx];

                                    // --- Store the corresponding p_pos and p_neg into the array Pos_neg_index_second ---
                                    if (Pos_neg_index_second[idx1] == NULL) {
                                        
                                        Pos_neg_index_second[idx1] = (int *)malloc(1 * sizeof(int));
                                        Pos_neg_index_second_col_sizes[idx1] = 1;
                                        if (Pos_neg_index_second[idx1] == NULL) {
                                            printf("Memory allocation failed for Pos_neg_index_second[%d]\n", idx1);
                                            return 1;
                                        }
                                    } else {
                                        int current_col_size = Pos_neg_index_second_col_sizes[idx1];
                                        Pos_neg_index_second[idx1] = (int *)realloc(Pos_neg_index_second[idx1], current_col_size * sizeof(int));
                                        if (Pos_neg_index_second[idx1] == NULL) {
                                            printf("Memory reallocation failed for Pos_neg_index_second[%d]\n", idx1);
                                            return 1;
                                        }
                                    }

                                    current_col_size = Pos_neg_index_second_col_sizes[idx1];
                                    Pos_neg_index_second[idx1][current_col_size - 1] = param_value_to_fill;
                                }

                                oracle_response_2_size++;

                                printf("Contents of oracle_response_2 (Column-wise): oracle_response_2_size is %d\n", oracle_response_2_size);
                                for (int col = 0; col < oracle_response_2_capacity; col++) {
                                    printf("Column %d: ", col);
                                    for (int row = 0; row < oracle_response_2_col_sizes[col]; row++) {
                                        printf("%d ", oracle_response_2[col][row]);
                                    }
                                    printf("\n");
                                }

                                printf("Contents of Pos_neg_index_second (Column-wise):\n");
                                for (int col = 0; col < Pos_neg_index_second_capacity; col++) {
                                    printf("Column %d: ", col);
                                    for (int row = 0; row < Pos_neg_index_second_col_sizes[col]; row++) {
                                        printf("%d ", Pos_neg_index_second[col][row]);
                                    }
                                    printf("\n");
                                }

                                // Iterate over the data in oracle_response_2, check, and store the relevant information into the corresponding arrays
                                for (int col = 0; col < MAX_ROWS; col++) 
                                {
                                    if (oracle_response_2_col_sizes[col] > 0) 
                                    {
                                        int val_0 = oracle_response_2[col][0];  

                                        if ((current_tree_state == 1 && val_0 == 0) || (current_tree_state == 2 && val_0 == -1)) {
                                           
                                            updated_sec_index_second_part = (int **)realloc(updated_sec_index_second_part, (updated_array_size_2 + 1) * sizeof(int *));
                                            if (updated_sec_index_second_part == NULL) {
                                                printf("Memory allocation failed during resizing.\n");
                                                return 1;
                                            }

                                            updated_sec_index_second_part[updated_array_size_2] = (int *)malloc(2 * sizeof(int));
                                            if (updated_sec_index_second_part[updated_array_size_2] == NULL) {
                                                printf("Memory allocation failed for updated_sec_index_second_part[%d]\n", updated_array_size_2);
                                                return 1;
                                            }

                                            updated_check1_array_2 = (int *)realloc(updated_check1_array_2, (updated_array_size_2 + 1) * sizeof(int));
                                            updated_index_array_2 = (int *)realloc(updated_index_array_2, (updated_index_array_size_2 + 1) * sizeof(int));  

                                            if (updated_check1_array_2 == NULL || updated_index_array_2 == NULL) {
                                                printf("Memory allocation failed during resizing.\n");
                                                return 1;
                                            }

                                            updated_sec_index_second_part[updated_array_size_2][0] = sec_index_second_part[col][0];  
                                            updated_sec_index_second_part[updated_array_size_2][1] = sec_index_second_part[col][1];  
                                            updated_check1_array_2[updated_array_size_2] = 1;

                                            updated_index_array_2[updated_index_array_size_2] = col;
                                            updated_index_array_size_2++;
                                            updated_array_size_2++;  
                                        }
                                        else if ((current_tree_state == 1 && val_0 == -1) || (current_tree_state == 2 && val_0 == 0)) {
                                           
                                            new_sec_index_second_part = (int **)realloc(new_sec_index_second_part, (new_array_size_2 + 1) * sizeof(int *));
                                            if (new_sec_index_second_part == NULL) {
                                                printf("Memory allocation failed during resizing.\n");
                                                return 1;
                                            }

                                            new_sec_index_second_part[new_array_size_2] = (int *)malloc(2 * sizeof(int));
                                            if (new_sec_index_second_part[new_array_size_2] == NULL) {
                                                printf("Memory allocation failed for new_sec_index_second_part[%d]\n", new_array_size_2);
                                                return 1;
                                            }

                                            new_check1_array_2 = (int *)realloc(new_check1_array_2, (new_array_size_2 + 1) * sizeof(int));
                                            index_array_2 = (int *)realloc(index_array_2, (index_array_size_2 + 1) * sizeof(int));  

                                            if (new_check1_array_2 == NULL || index_array_2 == NULL) {
                                                printf("Memory allocation failed during resizing.\n");
                                                return 1;
                                            }

                                            new_sec_index_second_part[new_array_size_2][0] = sec_index_second_part[col][0];  
                                            new_sec_index_second_part[new_array_size_2][1] = sec_index_second_part[col][1];  
                                            if(current_tree_state == 1)
                                            {
                                                new_check1_array_2[new_array_size_2] = 0;
                                                current_state = 1;
                                            }
                                            if(current_tree_state == 2)
                                            {
                                                new_check1_array_2[new_array_size_2] = 1;
                                                current_state = 2;
                                            }
                                           
                                            index_array_2[index_array_size_2] = col;
                                            index_array_size_2++;
                                            new_array_size_2++;  
                                        }
                                    }
                                }

                            
                                
                                for (int i = 0; i < updated_array_size_2; i++) {
                                    sec_index_second_part[i][0] = updated_sec_index_second_part[i][0];  
                                    sec_index_second_part[i][1] = updated_sec_index_second_part[i][1];  
                                }
                                printf("Updated sec_index_second_part contents after memcpy:\n");
                                for (int i = 0; i < updated_array_size_2; i++) {
                                    printf("sec_index_second_part[%d][0] = %d, sec_index_second_part[%d][1] = %d\n", 
                                        i, sec_index_second_part[i][0], i, sec_index_second_part[i][1]);
                                }

                                
                                if (updated_check1_array_2 != NULL) 
                                {
                                    
                                    int *temp_updated_check1_array_2_copy = (int *)malloc(updated_array_size_2 * sizeof(int));
                                    if (temp_updated_check1_array_2_copy == NULL) {
                                        printf("Memory allocation failed for temp_updated_check1_array_2_copy.\n");
                                        return 1;
                                    }

                                    memcpy(temp_updated_check1_array_2_copy, updated_check1_array_2, updated_array_size_2 * sizeof(int));

                                    check1_array = temp_updated_check1_array_2_copy;

                                    free(updated_check1_array_2);
                                    updated_check1_array_2 = NULL;
                                } else {
                                    printf("updated_check1_array_2 is NULL, using backup data.\n");
                                }

                                start_idx = 0;
                                end_idx = updated_array_size_2;
                                array_size = updated_array_size_2;  
                              
                                free(get_er_decrypt_array_all);
                                get_er_decrypt_array_all = (int *)malloc(array_size * sizeof(int));
                                if (get_er_decrypt_array_all == NULL) {
                                    printf("Memory allocation failed array_sizeis %d\n",array_size);
                                    return 1;
                                }
                                
                                get_oracle_response(ct, ss, ss1, pk, sk,
                                                    sec_index_second_part, 
                                                    get_er_decrypt_array_all, check1_array, 
                                                    param_list1_index, param_list2_index, 
                                                    start_idx, end_idx, 
                                                    &overall_oracle_count);
                                
                                printf("For new sec_index, actual oracle response is...\n");
                                for (int i = 0; i < array_size; i++) {
                                    printf("%d,", get_er_decrypt_array_all[i]);
                                }
                                printf("\n");

                                param_value_to_fill = 0;  
                                for (int i = 0; i < updated_array_size_2; i++) 
                                {
                                    int target_index = updated_index_array_2[i];  

                                    if (oracle_response_2[target_index] == NULL) {
                                        oracle_response_2[target_index] = (int *)malloc(1 * sizeof(int));  
                                        oracle_response_2_col_sizes[target_index] = 1;  
                                    } else {
                                        
                                        int current_col_size = oracle_response_2_col_sizes[target_index];
                                        oracle_response_2[target_index] = (int *)realloc(oracle_response_2[target_index], (current_col_size + 1) * sizeof(int));
                                        if (oracle_response_2[target_index] == NULL) {
                                            printf("Memory reallocation failed for oracle_response_2[%d]\n", target_index);
                                            return 1;
                                        }
                                        oracle_response_2_col_sizes[target_index] = current_col_size + 1;  
                                    }

                                    int current_col_size = oracle_response_2_col_sizes[target_index];
                                    oracle_response_2[target_index][current_col_size - 1] = get_er_decrypt_array_all[i];

                                    if (Pos_neg_index_second[target_index] == NULL) {
                                        Pos_neg_index_second[target_index] = (int *)malloc(1 * sizeof(int));  
                                        Pos_neg_index_second_col_sizes[target_index] = 1;  
                                        if (Pos_neg_index_second[target_index] == NULL) {
                                            printf("Memory allocation failed for Pos_neg_index_second[%d]\n", target_index);
                                            return 1;
                                        }
                                    } else {
                                       
                                        int current_col_size = Pos_neg_index_second_col_sizes[target_index];
                                        Pos_neg_index_second[target_index] = (int *)realloc(Pos_neg_index_second[target_index], (current_col_size + 1) * sizeof(int));
                                        if (Pos_neg_index_second[target_index] == NULL) {
                                            printf("Memory reallocation failed for Pos_neg_index_second[%d]\n", target_index);
                                            return 1;
                                        }
                                        
                                        Pos_neg_index_second_col_sizes[target_index] = current_col_size + 1;
                                    }

                                    current_col_size = Pos_neg_index_second_col_sizes[target_index];
                                    Pos_neg_index_second[target_index][current_col_size - 1] = param_value_to_fill;
                                }

                               
                                printf("Contents of oracle_response_2 (Column-wise):\n");
                                for (int col = 0; col < MAX_ROWS; col++) {
                                    printf("Column %d: ", col);
                                    for (int row = 0; row < oracle_response_2_col_sizes[col]; row++) {
                                        printf("%d ", oracle_response_2[col][row]);
                                    }
                                    printf("\n");
                                }


                                // Restore backup_sec_index_second_part to sec_index_second_part

                                for (int i = 0; i < MAX_ROWS; i++) {
                                    memcpy(sec_index_second_part[i], backup_sec_index_second_part[i], 2 * sizeof(int));
                                }


                                printf("Restored sec_index_second_part contents:\n");
                                for (int i = 0; i < u_1_u_2; i++) {
                                    printf("sec_index_second_part[%d][0] = %d, sec_index_second_part[%d][1] = %d\n", 
                                        i, sec_index_second_part[i][0], i, sec_index_second_part[i][1]);
                                }

                                printf("her10 is %d\n", array_size);
                                zeros= 0;

                                for (int i = 0; i < updated_array_size_2; i++) 
                                {
                                   
                                    int target_index = updated_index_array_2[i];

                                    if (target_index >= MAX_ROWS || target_index < 0) {
                                        printf("Error: target_index %d is out of bounds\n", target_index);
                                        return 1;
                                    }

                                    if (oracle_response_2_col_sizes[target_index] < 2) {
                                        printf("Error: oracle_response_2 column %d has less than 2 rows\n", target_index);
                                        continue;  
                                    }

                                    
                                    int val_row_0 = oracle_response_2[target_index][0];
                                    int val_row_1 = oracle_response_2[target_index][1];

                                    if (current_tree_state == 1 && val_row_0 == 0 && val_row_1 == 0) {
                                        zeros++;  
                                    }                                
                                 
                                    if (current_tree_state == 2 && val_row_0 == -1 && val_row_1 == -1) {
                                        zeros++;  
                                    }
                                  
                                    if ((current_tree_state == 1 && val_row_0 == 0 && val_row_1 == -1) || 
                                        (current_tree_state == 2 && val_row_0 == -1 && val_row_1 == 0)) 
                                    {
                                        
                                        new_sec_index_second_part = (int **)realloc(new_sec_index_second_part, (new_array_size_2 + 1) * sizeof(int *));
                                        new_sec_index_second_part[new_array_size_2] = (int *)malloc(2 * sizeof(int));

                                        new_check1_array_2 = (int *)realloc(new_check1_array_2, (new_array_size_2 + 1) * sizeof(int));
                                        index_array_2 = (int *)realloc(index_array_2, (index_array_size_2 + 1) * sizeof(int));

                                        if (new_sec_index_second_part == NULL || new_check1_array_2 == NULL || index_array_2 == NULL) {
                                            printf("Memory allocation failed during resizing.\n");
                                            return 1;
                                        }

                                        new_sec_index_second_part[new_array_size_2][0] = sec_index_second_part[target_index][0];
                                        new_sec_index_second_part[new_array_size_2][1] = sec_index_second_part[target_index][1];
                                        
                                        if (current_tree_state == 1) 
                                        {
                                            new_check1_array_2[new_array_size_2] = 1;
                                        } else if (current_tree_state == 2) 
                                        {
                                            new_check1_array_2[new_array_size_2] = 0;
                                        }

                                        index_array_2[index_array_size_2] = target_index;

                                        index_array_size_2++;
                                        new_array_size_2++;
                                    }
                                }

                                printf("zeros: %d\n",zeros);                                                        
                                // If the obtained distinguisher seems abnormal, invert it and recreate the base ciphertext
                                if (zeros > 130 || zeros <80) 
                                {
                                                                            
                                    printf("Condition met, jumping to rej...\n");                                      
                                    
                                    goto rej;
                                }
                                else
                                {
                                    // Now compute the actual oracle response for (l_11, l_12, l_13) using the remaining (u1, u2) by querying the PC oracle
                                    start_idx = weight_list_k2;
                                    end_idx = u_1_u_2;
                                    array_size = end_idx - start_idx;
                                
                                    free(get_er_decrypt_array_all);
                                    get_er_decrypt_array_all = (int *)malloc(array_size * sizeof(int));
                                    if (get_er_decrypt_array_all == NULL) {
                                        printf("Memory allocation failed\n");
                                        return 1;
                                    }
                                   
                                    free(check1_array);  
                                    check1_array = (int *)malloc(end_idx * sizeof(int));  
                                    if (check1_array == NULL) {
                                        printf("Memory allocation failed for check1_array\n");
                                        return 1;
                                    }

                                    for (int i = 0; i < end_idx; i++) {
                                        check1_array[i] = 0;
                                    }
                                   
                                    get_oracle_response(ct, ss, ss1, pk, sk,
                                                        sec_index_second_part, 
                                                        get_er_decrypt_array_all, check1_array, 
                                                        param_list1_index, param_list2_index, 
                                                        start_idx, end_idx, 
                                                        &overall_oracle_count);
                                
                                    printf("For array_size i %d; new sec_index, actual oracle response is...\n",array_size);
                                    for (int i = 0; i < array_size; i++) {
                                        printf("%d,", get_er_decrypt_array_all[i]);
                                    }
                                    printf("\n");  
                                    
                                    param_value_to_fill = 0;  
                                    
                                    for (int idx1 = start_idx; idx1 < end_idx; idx1++) 
                                    {
                                        
                                        if (oracle_response_2[idx1] == NULL) {
                                           
                                            oracle_response_2[idx1] = (int *)malloc(1 * sizeof(int));  
                                            oracle_response_2_col_sizes[idx1] = 1;  
                                            if (oracle_response_2[idx1] == NULL) {
                                                printf("Memory allocation failed for oracle_response_2[%d]\n", idx1);
                                                return 1;
                                            }
                                        } else {
                                            
                                            oracle_response_2_col_sizes[idx1] = 1;
                                        }

                                        oracle_response_2[idx1][0] = get_er_decrypt_array_all[idx1 - start_idx];

                                    
                                        if (Pos_neg_index_second[idx1] == NULL) {
                                            
                                            Pos_neg_index_second[idx1] = (int *)malloc(1 * sizeof(int));  
                                            Pos_neg_index_second_col_sizes[idx1] = 1;  
                                            if (Pos_neg_index_second[idx1] == NULL) {
                                                printf("Memory allocation failed for Pos_neg_index_second[%d]\n", idx1);
                                                return 1;
                                            }
                                        } else {
                                            
                                            Pos_neg_index_second_col_sizes[idx1] = 1;
                                        }

                                    
                                        Pos_neg_index_second[idx1][0] = param_value_to_fill;
                                    }

                            
                                    for (int col = 0; col < oracle_response_2_capacity; col++) {
                                        printf("Column %d: ", col);
                                        for (int row = 0; row < oracle_response_2_col_sizes[col]; row++) {
                                            printf("%d ", oracle_response_2[col][row]);
                                        }
                                        printf("\n");
                                    }                                

                                    // Iterate over the data in oracle_response_2, check, and store the relevant information into the corresponding arrays
                                    for (int col = start_idx; col < oracle_response_2_capacity; col++) 
                                    {
                                       
                                        if (oracle_response_2_col_sizes[col] > 0) 
                                        {
                                            int val_0 = oracle_response_2[col][0]; 

                                            if ((current_tree_state == 1 && val_0 == 0) || (current_tree_state == 2 && val_0 == -1)) {
                                                
                                                filtered_sec_index_second_part = (int **)realloc(filtered_sec_index_second_part, (filtered_array_second_length + 1) * sizeof(int *));
                                                if (filtered_sec_index_second_part == NULL) {
                                                    printf("Memory allocation failed during resizing.\n");
                                                    return 1;
                                                }

                                                filtered_sec_index_second_part[filtered_array_second_length] = (int *)malloc(2 * sizeof(int));
                                                if (filtered_sec_index_second_part[filtered_array_second_length] == NULL) {
                                                    printf("Memory allocation failed for filtered_sec_index_second_part[%d]\n", filtered_array_second_length);
                                                    return 1;
                                                }

                                                filtered_check2_array = (int *)realloc(filtered_check2_array, (filtered_array_second_length + 1) * sizeof(int));
                                                filtered_position_second_array = (int *)realloc(filtered_position_second_array, (filtered_position_second_array_length + 1) * sizeof(int));
                                                if (filtered_check2_array == NULL || filtered_position_second_array == NULL) {
                                                    printf("Memory allocation failed during resizing.\n");
                                                    return 1;
                                                }

                                                filtered_sec_index_second_part[filtered_array_second_length][0] = sec_index_second_part[col][0];
                                                filtered_sec_index_second_part[filtered_array_second_length][1] = sec_index_second_part[col][1];
                                                filtered_check2_array[filtered_array_second_length] = 1; 
                                                filtered_position_second_array[filtered_position_second_array_length] = col;

                                                filtered_array_second_length++;
                                                filtered_position_second_array_length++;
                                            }
                                           
                                            else if ((current_tree_state == 1 && val_0 == -1) || (current_tree_state == 2 && val_0 == 0)) {
                                                
                                                new_sec_index_second_part = (int **)realloc(new_sec_index_second_part, (new_array_size_2 + 1) * sizeof(int *));
                                                if (new_sec_index_second_part == NULL) {
                                                    printf("Memory allocation failed during resizing.\n");
                                                    return 1;
                                                }

                                                new_sec_index_second_part[new_array_size_2] = (int *)malloc(2 * sizeof(int));
                                                if (new_sec_index_second_part[new_array_size_2] == NULL) {
                                                    printf("Memory allocation failed for new_sec_index_second_part[%d]\n", new_array_size_2);
                                                    return 1;
                                                }

                                                new_check1_array_2 = (int *)realloc(new_check1_array_2, (new_array_size_2 + 1) * sizeof(int));
                                                index_array_2 = (int *)realloc(index_array_2, (index_array_size_2 + 1) * sizeof(int));
                                                if (new_check1_array_2 == NULL || index_array_2 == NULL) {
                                                    printf("Memory allocation failed during resizing.\n");
                                                    return 1;
                                                }

                                                new_sec_index_second_part[new_array_size_2][0] = sec_index_second_part[col][0];
                                                new_sec_index_second_part[new_array_size_2][1] = sec_index_second_part[col][1];

                                                if (current_tree_state == 1)
                                                {
                                                    new_check1_array_2[new_array_size_2] = 0;
                                                }
                                                if (current_tree_state == 2)
                                                {
                                                    new_check1_array_2[new_array_size_2] = 1;
                                                }

                                                index_array_2[index_array_size_2] = col;
                                                index_array_size_2++;
                                                new_array_size_2++;
                                            }
                                        }
                                    }

                                  
                                    // Overwrite sec_index_second_part with the data from filtered_sec_index_second_part
                                    for (int i = 0; i < filtered_array_second_length; i++) {
                                        sec_index_second_part[i][0] = filtered_sec_index_second_part[i][0];  
                                        sec_index_second_part[i][1] = filtered_sec_index_second_part[i][1];  
                                    }

                                    printf("Updated sec_index_second_part contents after memcpy:\n");
                                    for (int i = 0; i < filtered_array_second_length; i++) {
                                        printf("sec_index_second_part[%d][0] = %d, sec_index_second_part[%d][1] = %d\n", 
                                            i, sec_index_second_part[i][0], i, sec_index_second_part[i][1]);
                                    }

                                    if (filtered_check2_array != NULL) 
                                    {
                                        
                                        int *temp_filtered_check2_array_copy = (int *)malloc(filtered_array_second_length * sizeof(int));
                                        if (temp_filtered_check2_array_copy == NULL) {
                                            printf("Memory allocation failed for temp_filtered_check2_array_copy.\n");
                                            return 1;
                                        }

                                        memcpy(temp_filtered_check2_array_copy, filtered_check2_array, filtered_array_second_length * sizeof(int));

                                        check1_array = temp_filtered_check2_array_copy;

                                        free(filtered_check2_array);
                                        filtered_check2_array = NULL;
                                    } 
                                    else 
                                    {                                        
                                        printf("filtered_check2_array is NULL, using backup data.\n");
                                    }

                                    start_idx = 0;
                                    end_idx = filtered_array_second_length;
                                    array_size = filtered_array_second_length;  
                                
                                    free(get_er_decrypt_array_all);
                                    get_er_decrypt_array_all = (int *)malloc(array_size * sizeof(int));
                                    if (get_er_decrypt_array_all == NULL) {
                                        printf("Memory allocation failed, array_size is %d\n", array_size);
                                        return 1;
                                    }
                                    
                                    get_oracle_response(ct, ss, ss1, pk, sk,
                                                        sec_index_second_part, 
                                                        get_er_decrypt_array_all, check1_array, 
                                                        param_list1_index, param_list2_index, 
                                                        start_idx, end_idx, 
                                                        &overall_oracle_count);
                                    printf("For new sec_index, actual oracle response is...\n");
                                    for (int i = 0; i < array_size; i++) {
                                        printf("%d,", get_er_decrypt_array_all[i]);
                                    }
                                    printf("\n");

                                    param_value_to_fill = 0;  


                                    for (int i = 0; i < array_size; i++) 
                                    {
                                        int target_index = filtered_position_second_array[i];  

                                       
                                        if (oracle_response_2[target_index] == NULL) {
                                            oracle_response_2[target_index] = (int *)malloc(1 * sizeof(int));  
                                            oracle_response_2_col_sizes[target_index] = 1;  
                                        } else {
                                            
                                            int current_col_size = oracle_response_2_col_sizes[target_index];
                                            oracle_response_2[target_index] = (int *)realloc(oracle_response_2[target_index], (current_col_size + 1) * sizeof(int));
                                            if (oracle_response_2[target_index] == NULL) {
                                                printf("Memory reallocation failed for oracle_response_2[%d]\n", target_index);
                                                return 1;
                                            }
                                            oracle_response_2_col_sizes[target_index] = current_col_size + 1;  
                                        }

                                        int current_col_size = oracle_response_2_col_sizes[target_index];
                                        oracle_response_2[target_index][current_col_size - 1] = get_er_decrypt_array_all[i];

                                        if (Pos_neg_index_second[target_index] == NULL) {
                                            Pos_neg_index_second[target_index] = (int *)malloc(1 * sizeof(int)); 
                                            Pos_neg_index_second_col_sizes[target_index] = 1;  
                                            if (Pos_neg_index_second[target_index] == NULL) {
                                                printf("Memory allocation failed for Pos_neg_index_second[%d]\n", target_index);
                                                return 1;
                                            }
                                        } else {
                                            
                                            int current_col_size = Pos_neg_index_second_col_sizes[target_index];
                                            Pos_neg_index_second[target_index] = (int *)realloc(Pos_neg_index_second[target_index], (current_col_size + 1) * sizeof(int));
                                            if (Pos_neg_index_second[target_index] == NULL) {
                                                printf("Memory reallocation failed for Pos_neg_index_second[%d]\n", target_index);
                                                return 1;
                                            }
                                          
                                            Pos_neg_index_second_col_sizes[target_index] = current_col_size + 1;
                                        }

                                        current_col_size = Pos_neg_index_second_col_sizes[target_index];
                                        Pos_neg_index_second[target_index][current_col_size - 1] = param_value_to_fill;
                                    }

                                    printf("Contents of oracle_response_2 (Column-wise):\n");
                                    for (int col = 0; col < MAX_ROWS; col++) {
                                        printf("Column %d: ", col);
                                        for (int row = 0; row < oracle_response_2_col_sizes[col]; row++) {
                                            printf("%d ", oracle_response_2[col][row]);
                                        }
                                        printf("\n");
                                    }


                                    // Restore backup_sec_index_second_part to sec_index_second_part
                                    for (int i = 0; i < MAX_ROWS; i++) {
                                        memcpy(sec_index_second_part[i], backup_sec_index_second_part[i], 2 * sizeof(int));
                                    }

                                    for (int i = 0; i < array_size; i++) 
                                    {
                                        
                                        int target_index = filtered_position_second_array[i];

                                        if (target_index >= MAX_ROWS || target_index < 0) {
                                            printf("Error: target_index %d is out of bounds\n", target_index);
                                            return 1;
                                        }

                                        if (oracle_response_2_col_sizes[target_index] < 2) {
                                            printf("Error: oracle_response_2 column %d has less than 2 rows\n", target_index);
                                            continue;  
                                        }

                                        int val_row_0 = oracle_response_2[target_index][0];
                                        int val_row_1 = oracle_response_2[target_index][1];

                                        if (current_tree_state == 1 && val_row_0 == 0 && val_row_1 == 0) {
                                            zeros++;  
                                        }                                
                                      
                                        if (current_tree_state == 2 && val_row_0 == -1 && val_row_1 == -1) {
                                            zeros++;  
                                        }
                                       
                                        if ((current_tree_state == 1 && val_row_0 == 0 && val_row_1 == -1) || 
                                            (current_tree_state == 2 && val_row_0 == -1 && val_row_1 == 0)) 
                                        {
                                           
                                            new_sec_index_second_part = (int **)realloc(new_sec_index_second_part, (new_array_size_2 + 1) * sizeof(int *));
                                            new_sec_index_second_part[new_array_size_2] = (int *)malloc(2 * sizeof(int));

                                            new_check1_array_2 = (int *)realloc(new_check1_array_2, (new_array_size_2 + 1) * sizeof(int));
                                            index_array_2 = (int *)realloc(index_array_2, (index_array_size_2 + 1) * sizeof(int));

                                            if (new_sec_index_second_part == NULL || new_check1_array_2 == NULL || index_array_2 == NULL) {
                                                printf("Memory allocation failed during resizing.\n");
                                                return 1;
                                            }

                                            new_sec_index_second_part[new_array_size_2][0] = sec_index_second_part[target_index][0];
                                            new_sec_index_second_part[new_array_size_2][1] = sec_index_second_part[target_index][1];
                                            
                                            if (current_tree_state == 1) 
                                            {
                                                new_check1_array_2[new_array_size_2] = 1;
                                            } else if (current_tree_state == 2) 
                                            {
                                                new_check1_array_2[new_array_size_2] = 0;
                                            }

                                            index_array_2[index_array_size_2] = target_index;

                                            index_array_size_2++;
                                            new_array_size_2++;
                                        }
                                    }


                                    use_new_sec_index = true;

                                    param_which_list = 1;
                                    param_list2_index = 0; 
                            
                                    if(current_tree_state == 1)
                                    {
                                        coding_arr_1[tree_size_1++] = create_node(true, 1);
                                        coding_arr_1[tree_size_1++] = create_node(false, -1);                           
                                        coding_arr_1[tree_size_1++] = create_node(true, 2);
                                        coding_arr_1[tree_size_1++] = NULL;
                                        coding_arr_1[tree_size_1++] = create_node(false, -2); 
                                                                        
                                        current_state = 1;   
                                    }
                                    if(current_tree_state == 2)
                                    {
                                        
                                        coding_arr_1[tree_size_1++] = create_node(true, 0);
                                        coding_arr_1[tree_size_1++] = create_node(false, -2);                           
                                        coding_arr_1[tree_size_1++] = create_node(false, -0);
                                        coding_arr_1[tree_size_1++] = NULL;
                                        coding_arr_1[tree_size_1++] = NULL;
                                        coding_arr_1[tree_size_1++] = create_node(true, 2);  
                                    

                                        current_state = 2; 
                                    }
                                    state_oracle_now++;
                                }    
                              
                            }     
                         
                            if (current_dis == 2)
                            {                                    
                               
                                                  
                                for (int i = 0; i < weight_list_k1; i++) 
                                {
                                    int target_index = index_array_2[i];  

                                    if (oracle_response_2[target_index] == NULL) {
                                        oracle_response_2[target_index] = (int *)malloc(1 * sizeof(int));  
                                        oracle_response_2_col_sizes[target_index] = 1;  
                                    } else {
                                        
                                        int current_col_size = oracle_response_2_col_sizes[target_index];
                                        oracle_response_2[target_index] = (int *)realloc(oracle_response_2[target_index], (current_col_size + 1) * sizeof(int));
                                        if (oracle_response_2[target_index] == NULL) {
                                            printf("Memory reallocation failed for oracle_response_2[%d]\n", target_index);
                                            return 1;
                                        }
                                        oracle_response_2_col_sizes[target_index] = current_col_size + 1;  
                                    }

                                    int current_col_size = oracle_response_2_col_sizes[target_index];
                                    oracle_response_2[target_index][current_col_size - 1] = get_er_decrypt_array_all[i];  
                                }

                                printf("Contents of oracle_response_2 (Column-wise):\n");
                                for (int col = 0; col < MAX_ROWS; col++) 
                                {
                                    printf("Column %d: ", col);
                                    for (int row = 0; row < oracle_response_2_col_sizes[col]; row++) 
                                    {
                                        printf("%d ", oracle_response_2[col][row]);
                                    }
                                    printf("\n");
                                }
                                param_value_to_fill = 0;
                                for (int i = 0; i < weight_list_k1; i++) 
                                {
                                    int target_index = index_array_2[i];  
                                    int current_col_size = Pos_neg_index_second_col_sizes[target_index];

                                    if (new_check1_array_2[i] == 0) {
                                        
                                        Pos_neg_index_second[target_index] = (int *)realloc(Pos_neg_index_second[target_index], (current_col_size + 1) * sizeof(int));
                                        if (Pos_neg_index_second[target_index] == NULL) {
                                            printf("Memory reallocation failed for Pos_neg_index_second[%d]\n", target_index);
                                            return 1;
                                        }
                                        Pos_neg_index_second[target_index][current_col_size] = param_value_to_fill;
                                    } 
                                    else if (new_check1_array_2[i] == 1) {
                                        
                                        Pos_neg_index_second[target_index] = (int *)realloc(Pos_neg_index_second[target_index], (current_col_size + 1) * sizeof(int));
                                        if (Pos_neg_index_second[target_index] == NULL) {
                                            printf("Memory reallocation failed for Pos_neg_index_second[%d]\n", target_index);
                                            return 1;
                                        }
                                        Pos_neg_index_second[target_index][current_col_size] = param_value_to_fill;
                                    }

                                    Pos_neg_index_second_col_sizes[target_index] = current_col_size + 1;
                                }

                                printf("Pos_neg_index_second, column-wise output:\n");
                                for (int col = 0; col < Pos_neg_index_second_capacity; col++) {
                                    printf("Column %d: ", col);
                                    for (int row = 0; row < Pos_neg_index_second_col_sizes[col]; row++) {
                                        printf("%d ", Pos_neg_index_second[col][row]);
                                    }
                                    printf("\n");
                                }

                                start_idx = weight_list_k1;
                                end_idx = new_array_size_2;
                                array_size = end_idx - start_idx;  
                                
                                free(get_er_decrypt_array_all);
                                get_er_decrypt_array_all = (int *)malloc(array_size * sizeof(int));
                                if (get_er_decrypt_array_all == NULL) {
                                    printf("Memory allocation failed array_sizeis %d\n",array_size);
                                    return 1;
                                }
                                
                                get_oracle_response(ct, ss, ss1, pk, sk,
                                                    sec_index_second_part, 
                                                    get_er_decrypt_array_all, check1_array, 
                                                    param_list1_index, param_list2_index, 
                                                    start_idx, end_idx, 
                                                    &overall_oracle_count);
                                
                                printf("For new sec_index, actual oracle response is...\n");
                                for (int i = 0; i < array_size; i++) {
                                    printf("%d,", get_er_decrypt_array_all[i]);
                                }
                                printf("\n");

                                for (int i = weight_list_k1; i < new_array_size_2; i++) 
                                {
                                    int target_index = index_array_2[i];  

                                    if (oracle_response_2[target_index] == NULL) {
                                        oracle_response_2[target_index] = (int *)malloc(1 * sizeof(int));  
                                        oracle_response_2_col_sizes[target_index] = 1;  
                                    } else {
                                        
                                        int current_col_size = oracle_response_2_col_sizes[target_index];
                                        oracle_response_2[target_index] = (int *)realloc(oracle_response_2[target_index], (current_col_size + 1) * sizeof(int));
                                        if (oracle_response_2[target_index] == NULL) {
                                            printf("Memory reallocation failed for oracle_response_2[%d]\n", target_index);
                                            return 1;
                                        }
                                        oracle_response_2_col_sizes[target_index] = current_col_size + 1;  
                                    }

                                    int current_col_size = oracle_response_2_col_sizes[target_index];
                                    oracle_response_2[target_index][current_col_size - 1] = get_er_decrypt_array_all[i-weight_list_k1];  
                                }

                                printf("Contents of oracle_response_2 (Column-wise):\n");
                                for (int col = 0; col < MAX_ROWS; col++) 
                                {
                                    printf("Column %d: ", col);
                                    for (int row = 0; row < oracle_response_2_col_sizes[col]; row++) 
                                    {
                                        printf("%d ", oracle_response_2[col][row]);
                                    }
                                    printf("\n");
                                }
                                param_value_to_fill = 0;
                                for (int i = weight_list_k1; i < new_array_size_2; i++) 
                                {
                                    int target_index = index_array_2[i];  
                                    int current_col_size = Pos_neg_index_second_col_sizes[target_index];

                                    if (new_check1_array_2[i] == 0) {
                                        
                                        Pos_neg_index_second[target_index] = (int *)realloc(Pos_neg_index_second[target_index], (current_col_size + 1) * sizeof(int));
                                        if (Pos_neg_index_second[target_index] == NULL) {
                                            printf("Memory reallocation failed for Pos_neg_index_second[%d]\n", target_index);
                                            return 1;
                                        }
                                        Pos_neg_index_second[target_index][current_col_size] = param_value_to_fill;
                                    } 
                                    else if (new_check1_array_2[i] == 1) {
                                        
                                        Pos_neg_index_second[target_index] = (int *)realloc(Pos_neg_index_second[target_index], (current_col_size + 1) * sizeof(int));
                                        if (Pos_neg_index_second[target_index] == NULL) {
                                            printf("Memory reallocation failed for Pos_neg_index_second[%d]\n", target_index);
                                            return 1;
                                        }
                                        Pos_neg_index_second[target_index][current_col_size] = param_value_to_fill;
                                    }

                                    Pos_neg_index_second_col_sizes[target_index] = current_col_size + 1;
                                }

                                printf("Pos_neg_index_second, column-wise output:\n");
                                for (int col = 0; col < Pos_neg_index_second_capacity; col++) {
                                    printf("Column %d: ", col);
                                    for (int row = 0; row < Pos_neg_index_second_col_sizes[col]; row++) {
                                        printf("%d ", Pos_neg_index_second[col][row]);
                                    }
                                    printf("\n");
                                }

                                for (int i = 0; i < MAX_ROWS; i++) {
                                    memcpy(sec_index_second_part[i], backup_sec_index_second_part[i], 2 * sizeof(int));
                                }
                                   

                                state_oracle_now++; 

                            }
                        }
                        else
                        {   
                                                           
                            if (current_state == 0)
                            {                               
                                param_which_list=0;
                                param_list1_index++;   
                            }
                            if (current_state == 1)
                            {
                                param_which_list=1;
                                param_list2_index++;   
                            }
                            if (current_state == 2)
                            {
                                param_which_list=1;
                                param_list2_index++;    
                            }                                         
                           
                        }
                        if (state_oracle_now == 2)
                        {
                            break;
                        }
                        
                        if (param_list1_index >= 3 || param_list2_index >= 3) 
                        {                        
                            printf("Unsuccessful search, jumping to rej...\n");                            
                            goto rej;
                           
                        }
                       
                    }
 
                    // Store the parameters of the attack ciphertext used

                    FILE *actual_oracle_response_for_param_list_file = fopen("actual_oracle_response_for_param_list.bin", "a");
                    if (actual_oracle_response_for_param_list_file == NULL) {
                        printf("Failed to open file: actual_oracle_response_for_param_list.bin\n");
                        return 1;
                    }
                        
                    int indices[4]; 
                    indices[0] = 0; indices[1] = 1;
                    
                    fprintf(actual_oracle_response_for_param_list_file, "for NO_TESTS is %d single_used_param:\n", pq);
                    
                    for (int i = 0; i < 2; i++) {  
                        int index = indices[i];
                        fprintf(actual_oracle_response_for_param_list_file, "Record %d: param_which_list = %d, param_list1_index = %d, param_list2_index = %d\n",
                                index,
                                param_records[index].param_which_list,
                                param_records[index].param_list1_index,
                                param_records[index].param_list2_index);
                    }
                
                    fclose(actual_oracle_response_for_param_list_file);

                    // Store the obtained internal encoding list into the actual_match_table file
                    FILE *actual_match_table_file = fopen("actual_match_table.bin", "a");
                    if (actual_match_table_file == NULL) {
                        printf("Failed to open file: actual_match_table.bin\n");
                        return 1;
                    }
                    fprintf(actual_match_table_file, "for NO_TESTS is %d tree each test:\n", pq);

                    Node *coding_tree_1 = recursive_tree_from_array(coding_arr_1, 0, tree_size_1);
                    SimpleOracle *oracle = create_simple_oracle(1.0);  // Perfect accuracy

                    printf("Chosen encoding1\n");
                    fprintf(actual_match_table_file, "tree_1:\n");  

                    int out[MAX_BIT_LENGTH];  // Array to store output
                    int size;                 // To store the size of the output array
                    int sum_weight = 4;       // Range for s from -sum_weight to sum_weight

                    for (int s = -sum_weight; s <= sum_weight; s++) 
                    {
                        // Sample the coefficients using the SimpleOracle and the current tree
                        sample_coef_with_simple_oracle(oracle, s, coding_tree_1, out, &size);

                        // Output the result to the console
                        printf("%d \t| (", s);
                        fprintf(actual_match_table_file, "%d \t| (", s);  

                        for (int i = 0; i < size; i++) {
                            printf("%d", out[i]);
                            fprintf(actual_match_table_file, "%d", out[i]);  

                            if (i < size - 1) {
                                printf(", ");
                                fprintf(actual_match_table_file, ", ");  
                            }
                        }
                        printf(")\n");
                        fprintf(actual_match_table_file, ")\n");  
                    }

                    fclose(actual_match_table_file);

                   // For inspection purposes only, save the actual oracle response(s) to a file
                    FILE *actual_oracle_response_file = fopen("actual_oracle_response.txt", "a");
                    if (actual_oracle_response_file == NULL) {
                        printf("Failed to open file: actual_oracle_response.txt\n");
                        return 1;
                    }

                    fprintf(actual_oracle_response_file, "for NO_TESTS is %d cond_prob_loop is %d actual oracle response for each test:\n", pq, cond_prob_loop);

                    int actual_loop_count = 1;
                    
                    for (int col = 0; col < u_1_u_2; col++) {
                        fprintf(actual_oracle_response_file, "actual_loop_count:%d, actual_oracle_response:", actual_loop_count++);
                        
                        for (int row = 0; row < oracle_response_2_col_sizes[col]; row++) {
                            fprintf(actual_oracle_response_file, "%d ", oracle_response_2[col][row]);
                        }
                        fprintf(actual_oracle_response_file, "\n");
                    }

                    fclose(actual_oracle_response_file);

               
                    int beta_u_counts[5] = {0}; 
                    //char filename[256]; 
                    int theoretical_oracle_value[table_column_size];
                    int beta_u;
                   
                    char beta_u_expr[256] = ""; 
                    int coeff_now =0;
                    int coeff_now_1, coeff_now_2;
                   
                    theoretical_loop_count = 0; 
                   
                    // Restore backup_sec_index_second_part to sec_index_second_part
                    for (int i = 0; i < MAX_ROWS; i++) {
                        memcpy(sec_index_second_part[i], backup_sec_index_second_part[i], 2 * sizeof(int));
                    }

 
                    // For inspection purposes only, save the index list U1 = (u1, u2) used to a file
                    FILE *u_1_u_2_logs_file = fopen("u_1_u_2_logs.txt", "a");
                    if (u_1_u_2_logs_file == NULL) {
                        printf("Failed to open file: u_1_u_2_logs.txt\n");
                        return 1;
                    }

                    fprintf(u_1_u_2_logs_file, "for NO_TESTS is %d cond_prob_loop is %d u_1_u_2 for each test:\n", pq, cond_prob_loop);

                    int List_count = 1;

                    for (int i = 0; i < u_1_u_2; i++) {
                        fprintf(u_1_u_2_logs_file, "List_count:%d, [%d, %d]\n", List_count, sec_index_second_part[i][0], sec_index_second_part[i][1]);
                        List_count++;
                    }

                    fclose(u_1_u_2_logs_file);  
                    
                    // For inspection purposes only, save the theoretical oracle response to a file

                    beta_u_and_theoretical_oracle_response_file = fopen("beta_u_and_theoretical_oracle_response.bin", "a");
                    if (beta_u_and_theoretical_oracle_response_file == NULL) {
                        printf("Failed to open file: beta_u_and_theoretical_oracle_response.bin\n");
                        return 1;
                    }
                   
                    fprintf(beta_u_and_theoretical_oracle_response_file, "for NO_TESTS is %d cond_prob_loop is %d beta_u and theoretical oracle response for each test:\n",pq, cond_prob_loop);

                    fclose(beta_u_and_theoretical_oracle_response_file);
                    // For inspection purposes only, compute the theoretical oracle response
                    for (int idx1 = 0; idx1 < u_1_u_2; idx1++) 
                    {
                        
                        theoretical_loop_count++; 
                        beta_u = 0;
                        beta_u_expr[0] = '\0'; 
                        int beta_coeff_count = 0; 
                        
                        for(int j = 0; j < 2; j++) 
                        {
                            int uj = sec_index_second_part[idx1][j];
                            printf("third  uj is %d\n", uj);
                            char term[64]; 
                            if(collision_index > uj) {
                                int coeff_now = collision_index - uj;
                                beta_u += global_f[coeff_now];
                                snprintf(term, sizeof(term), "%d", coeff_now);
                            } else {
                                int coeff_now_1 = (p - uj + collision_index - 1) % p;
                                int coeff_now_2 = (p - uj + collision_index) % p;
                                beta_u += global_f[coeff_now_1] + global_f[coeff_now_2];
                                snprintf(term, sizeof(term), "%d,%d", coeff_now_1, coeff_now_2);
                            }

                            if (j > 0) {
                                strcat(beta_u_expr, ",");
                            }
                            strcat(beta_u_expr, term);  
                        }

                        if (collision_value == 1) {
                            beta_u = -beta_u;
                        }

                        sample_coef_with_simple_oracle(oracle, beta_u, coding_tree_1, out, &size);  // 传递 beta_u 作为 s

                        printf("beta_u: %d, theoretical_oracle_output: (", beta_u);
                        for (int i = 0; i < size; i++) {
                            printf("%d", out[i]);
                            if (i < size - 1) {
                                printf(", ");
                            }
                        }
                        printf(")\n");

                        beta_u_and_theoretical_oracle_response_file = fopen("beta_u_and_theoretical_oracle_response.bin", "a");
                        if (beta_u_and_theoretical_oracle_response_file == NULL) {
                            printf("Failed to open file: beta_u_and_theoretical_oracle_response.bin\n");
                            return 1;
                        }

                        fprintf(beta_u_and_theoretical_oracle_response_file, "theoretical_loop_count:%d, beta_u:%d, theoretical_oracle_value:", theoretical_loop_count, beta_u);

                        for (int i = 0; i < size; i++) 
                        {
                            fprintf(beta_u_and_theoretical_oracle_response_file, "%d", out[i]);
                            if (i < size - 1) {
                                fprintf(beta_u_and_theoretical_oracle_response_file, " ");
                            }
                        }
                        fprintf(beta_u_and_theoretical_oracle_response_file, "\n");
                        fclose(beta_u_and_theoretical_oracle_response_file);

                    }    
                    // Now compute the conditional probability based on the actual oracle response and p_pos/p_neg, 
                    // and provide it as input to the LDPC decoder

                    int w1 = 288;
                    long double Pr[new_table_row_size];
                    FILE *file1 = NULL;
                    FILE *file2 = NULL;
                    
                    char command[512];
                    // Updated usage
                    long double **oracle_accuracy = init_oracle_accuracy();
                    FalsePositiveNegativePositionalOracle *pr_oracle = create_fp_fn_oracle(oracle_accuracy, 1);   

                    for (int coll_index = collision_index; coll_index < collision_index + 1; coll_index++)
                    {     
                        char filename[256];
                        #if (DO_PRINT == 1)
                        // Construct filename
                        sprintf(filename, "When 1 %d for alpha_u_and_conditional_probabilities.bin", coll_index);
                        file1 = fopen(filename, "w");
                        if (file1 == NULL) {
                            printf("Failed to open file: %s\n", filename);
                            continue;
                        }

                        #endif
                        
                        char beta_u_expr[256] = "";

                        int coeff_now_att;
                        int coeff_now_1_att, coeff_now_2_att;
                        
                        for (int idx1 = 0; idx1 < u_1_u_2; idx1++) 
                        {
                            
                            beta_u_expr[0] = '\0';  
                            int beta_coeff_count = 0; 
                           
                            for(int j = 0; j < 2; j++) 
                            {
                                int uj = sec_index_second_part[idx1][j];
                                char term[64]; 
                                if (coll_index > uj) 
                                {
                                    coeff_now_att = coll_index - uj;
                                    snprintf(term, sizeof(term), "%d", coeff_now_att);
                                    beta_coeff_count++;
                                } 
                                else 
                                {
                                    coeff_now_1_att = (p - uj + coll_index - 1) % (p);
                                    coeff_now_2_att = (p - uj + coll_index) % (p);
                                    snprintf(term, sizeof(term), "%d,%d", coeff_now_1_att, coeff_now_2_att);
                                    beta_coeff_count += 2;
                                }

                                if (j > 0) 
                                {
                                    strcat(beta_u_expr, ",");
                                }
                                strcat(beta_u_expr, term);
                            }


                            calculate_beta_probabilities(Pr, beta_coeff_count);

                            int *y = NULL;
                            int y_len = 0;

                            // **Find the size of column idx1**
                            // Determine the length of column idx1 using oracle_response_2_col_sizes[idx1]

                            if (oracle_response_2[idx1] != NULL) 
                            {
                                
                                y_len = oracle_response_2_col_sizes[idx1];  

                                y = (int *)malloc(y_len * sizeof(int));
                                if (y == NULL) 
                                {
                                    printf("Memory allocation failed for y\n");
                                    return 1;
                                }

                                for (int i = 0; i < y_len; i++) 
                                {
                                    y[i] = oracle_response_2[idx1][i];
                                }
                                
                                
                            } 
                            else 
                            {
                                printf("No data in oracle_response_2[%d]\n", idx1);  
                            }

                            int *y_prob = NULL;
                            int y_prob_len = 0;

                            // **Find the size of column idx1**
                            // Determine the length of column idx1 using Pos_neg_index_second_col_sizes[idx1]

                            if (Pos_neg_index_second[idx1] != NULL) 
                            {
                               
                                y_prob_len = Pos_neg_index_second_col_sizes[idx1];  

                                y_prob = (int *)malloc(y_prob_len * sizeof(int));
                                if (y_prob == NULL) 
                                {
                                    printf("Memory allocation failed for y_prob\n");
                                    return 1;
                                }

                                for (int i = 0; i < y_prob_len; i++) 
                                {
                                    y_prob[i] = Pos_neg_index_second[idx1][i];
                                }                                
                            } 
                            else 
                            {
                                printf("No data in Pos_neg_index_second[%d]\n", idx1);  
                            }
                            long double conditional_probabilities_2[new_table_row_size] = {0.0};
                          
                            if (y != NULL && y_prob != NULL) 
                            {
                                for (int s = -sum_weight; s <= sum_weight; s++) 
                                {
                                    long double pr_x_y = pr_cond_xy_adaptive(s, y, y_len, pr_oracle, Pr, sum_weight, coding_tree_1, y_prob);  
                                    conditional_probabilities_2[s + sum_weight] = pr_x_y;  
                                }
                            }

                            if (y != NULL) 
                            {
                                free(y);
                            }
                            if (y_prob != NULL) 
                            {
                                free(y_prob);
                            }

                            #if (DO_PRINT == 1)

                            FILE *file4 = file1;  

                            fprintf(file4, "%s\n", beta_u_expr);

                            for (int i = 0; i < new_table_row_size; i++) 
                            {
                                if (i > 0) 
                                {
                                    fprintf(file4, ","); 
                                }
                                fprintf(file4, "%.20Lf", conditional_probabilities_2[i]);
                            }
                            fprintf(file4, "\n");


                            #endif
                        }
                    
                        #if (DO_PRINT == 1)
                        if (file1 != NULL) fclose(file1);
                    
                        #endif
                        // Call the LDPC decoder and store the output into files such as ldpc_fprime_output_file
                         
                        FILE *after_change_ldpc_output_file = fopen("after_change_ldpc.bin", "a");
                        if (after_change_ldpc_output_file == NULL) {
                            printf("Failed to open file: after_change_ldpc.bin\n");
                            return 1;
                        }
                       
                        fprintf(after_change_ldpc_output_file, "pq_counter: %d,cond_prob_loop: %d,collision_index: %d,collision_value: %d,", pq, cond_prob_loop, collision_index, collision_value);
                        fprintf(after_change_ldpc_output_file, "\n");
                       
                        fclose(after_change_ldpc_output_file);

                        char command[1024];  

                        snprintf(command, sizeof(command), 
                        "/home/fei/C/SCA_Assisted_CCA_on_NTRU/NTRU_Prime/PC_Oracle_based_SCA/Attack_Simulations/sntrup761/SCA-LDPC/python-virtualenv/bin/python3 "
                        "/home/fei/C/SCA_Assisted_CCA_on_NTRU/NTRU_Prime/PC_Oracle_based_SCA/Attack_Simulations/sntrup761/SCA-LDPC/simulate-with-python/ldpc_decode.py %d %d", 
                        coll_index, pq);
                        
                        printf("Executing command: %s\n", command);
                    
                        int ret = system(command);  
                        if (ret == -1) {
                            perror("Error executing Python script");
                            return EXIT_FAILURE;
                        }
                    
                        printf("Python script executed successfully.\n");
                        printf("end\n");

                        // ================= Read data from the fprime_weight.txt file ===================
                        // Read the weight of the LDPC output secret key

                        FILE *weight_file = fopen(OUTPUT_WEIGHT_FILENAME, "r");
                        if (weight_file == NULL) {
                            printf("Failed to open file: %s\n", OUTPUT_WEIGHT_FILENAME);
                            return 1;
                        }

                        char weight_buffer[100];  
                        int weight_value;
                        int is_skipping = 0;
                        int is_file_empty = 0;

                        fseek(weight_file, 0, SEEK_END);
                        long filesize = ftell(weight_file);
                        rewind(weight_file);

                        if (filesize == 0) {
                            is_file_empty = 1;
                        } else if (fscanf(weight_file, "%99s", weight_buffer) != 1) {
                            printf("Failed to read from file: %s\n", OUTPUT_WEIGHT_FILENAME);
                            fclose(weight_file);
                            return 1;
                        }

                        if (!is_file_empty) {
                            if (strcmp(weight_buffer, "skipping") == 0) {
                                is_skipping = 1;
                            } else {
                                
                                if (sscanf(weight_buffer, "%d", &weight_value) != 1) {
                                    printf("Invalid integer in file: %s (read: %s)\n", OUTPUT_WEIGHT_FILENAME, weight_buffer);
                                    fclose(weight_file);
                                    return 1;
                                }
                            }
                        }

                        fclose(weight_file);

                        // ================ Write the read values to secret_key_weight.txt ================
                        FILE *secret_weight_file = fopen("secret_key_weight.txt", "a");
                        if (secret_weight_file == NULL) {
                            printf("Failed to open file: secret_key_weight.txt\n");
                            return 1;
                        }

                        fprintf(secret_weight_file, "for NO_TESTS is %d Assuming collision index is %d\n", pq, coll_index);

                        if (is_file_empty) {
                            fprintf(secret_weight_file, "NULL\n");
                            printf("File is empty; wrote NULL to secret_key_weight.txt.\n");
                        } else if (is_skipping) {
                            fprintf(secret_weight_file, "skipping\n");
                            printf("File contained 'skipping'; wrote skipping to secret_key_weight.txt.\n");
                        } else {
                            fprintf(secret_weight_file, "%d\n", weight_value);
                            printf("Weight value %d written to secret_key_weight.txt successfully.\n", weight_value);
                        }

                        fclose(secret_weight_file);
                        printf("end\n");

                        // ================ Store the average number of errors in the recovered secret key for each test ================

                        FILE *ldpc_input_file = fopen(OUTPUT_LDPC_PATH, "r");
                        if (ldpc_input_file == NULL) {
                            printf("Failed to open file: %s\n", OUTPUT_LDPC_PATH);
                            return 1;
                        }
                        
                        FILE *ldpc_final_output_file = fopen("ldpc_final_output.txt", "a");
                        if (ldpc_final_output_file == NULL) {
                            printf("Failed to open file: ldpc_final_output.txt\n");
                            fclose(ldpc_input_file);
                            return 1;
                        }
                        
                        fprintf(ldpc_final_output_file,
                                "for NO_TESTS is %d cond_prob_loop is %d Assuming collision index is %d\n",
                                pq, cond_prob_loop, coll_index);
                        
                        fseek(ldpc_input_file, 0, SEEK_END);
                        long fsize = ftell(ldpc_input_file);
                        rewind(ldpc_input_file);
                        
                        if (fsize == 0) {
                            printf("Warning: LDPC input file is empty.\n");
                        }
                        
                        char *full_buffer = (char *)malloc(fsize + 1);
                        if (full_buffer == NULL) {
                            printf("Memory allocation failed.\n");
                            fclose(ldpc_input_file);
                            fclose(ldpc_final_output_file);
                            return 1;
                        }
                        
                        size_t read_bytes = fread(full_buffer, 1, fsize, ldpc_input_file);
                        if (read_bytes != fsize) {
                            printf("Warning: Only %zu of %ld bytes read from %s.\n", read_bytes, fsize, OUTPUT_LDPC_PATH);
                        }
                        
                        full_buffer[read_bytes] = '\0';  
                        
                        fwrite(full_buffer, 1, read_bytes, ldpc_final_output_file);
                        
                        free(full_buffer);
                        fclose(ldpc_input_file);
                        fclose(ldpc_final_output_file);
                        
                        printf("LDPC output successfully copied to ldpc_final_output.txt.\n");
                        
                        // ================== Store the LDPC output secret key ==================
                        FILE *fprime_file = fopen(OUTPUT_FPRIME_PATH, "r");
                        if (fprime_file == NULL) {
                            printf("Failed to open file: %s\n", OUTPUT_FPRIME_PATH);
                            return 1;
                        }

                        FILE *ldpc_fprime_output_file = fopen("ldpc_fprime_output.txt", "a");
                        if (ldpc_fprime_output_file == NULL) {
                            printf("Failed to open file: ldpc_fprime_output.txt\n");
                            fclose(fprime_file);
                            return 1;
                        }

                        fprintf(ldpc_fprime_output_file, "For NO_TESTS is %d:\n", pq);

                       
                        fseek(fprime_file, 0, SEEK_END);
                        long fprime_fsize = ftell(fprime_file);
                        rewind(fprime_file);

                        if (fprime_fsize == 0) {
                            printf("Warning: fprime_output.txt is empty.\n");
                            fprintf(ldpc_fprime_output_file, "(empty)\n");
                        } else {
                            char buffer[4096];  
                            int read_success = 0;
                            int first = 1;

                            if (fgets(buffer, sizeof(buffer), fprime_file) != NULL) {
                                char *token = strtok(buffer, ",");
                                while (token != NULL) {
                                    int value;
                                    if (sscanf(token, "%d", &value) == 1) {
                                        read_success = 1;
                                        if (!first) {
                                            fprintf(ldpc_fprime_output_file, ",");
                                        }
                                        fprintf(ldpc_fprime_output_file, "%d", value);
                                        first = 0;
                                    }
                                    token = strtok(NULL, ",");
                                }
                            }

                            if (!read_success) {
                                printf("Warning: No valid integers found in %s.\n", OUTPUT_FPRIME_PATH);
                                fprintf(ldpc_fprime_output_file, "(no valid integers)\n");
                            } else {
                                fprintf(ldpc_fprime_output_file, "\n");
                            }
                        }

                        fclose(fprime_file);
                        fclose(ldpc_fprime_output_file);

                        printf("fprime_output successfully processed to ldpc_fprime_output.txt.\n");

                    }
                    
                   cond_prob_loop++; 
                } 
              
            
                // Store the oracle count for each test
                FILE *overall_oracle_calls_count_file = fopen("Overall_oracle_calls_count.txt", "a");
                if (overall_oracle_calls_count_file == NULL)
                {
                    printf("Failed to open file: Overall_oracle_calls_count.txt\n");
                    return 1;
                }

                fprintf(overall_oracle_calls_count_file, "for NO_TESTS is %d cond_prob_loop is %d oracle count for each test:\n", pq, cond_prob_loop - 1);
                fprintf(overall_oracle_calls_count_file, "overall_oracle_count is %d\n", overall_oracle_count);

                fclose(overall_oracle_calls_count_file);

                printf("Reached Here...\n");
               
                #else
                successful_attack_done = successful_attack_done + 1;
                #endif

                if (oracle_response_2 != NULL) {  
                    for (int i = 0; i < oracle_response_2_size; i++) {
                        if (oracle_response_2[i] != NULL) {  
                            free(oracle_response_2[i]);  
                            oracle_response_2[i] = NULL;  
                        }
                    }
                    free(oracle_response_2);  
                    oracle_response_2 = NULL;  
                }

                if (oracle_response_2_col_sizes != NULL) {  
                    free(oracle_response_2_col_sizes);  
                    oracle_response_2_col_sizes = NULL;  
                }

                if (param_records != NULL) {
                    free(param_records);
                    param_records = NULL;  
                }

                if (coding_arr_1 != NULL) {
                    free(coding_arr_1);
                    coding_arr_1 = NULL;  
                }
    }  

    for (int i = 0; i < MAX_ROWS; i++) {
        free(sec_index_second_part[i]);
        free(backup_sec_index_second_part[i]);
    }
    free(sec_index_second_part);
    free(backup_sec_index_second_part);
    free(c_value_for_attack_1_1);
    free(c_value_for_attack_1_2);
    free(c_value_for_attack_1_3);

    #endif

    return KAT_SUCCESS;
}
