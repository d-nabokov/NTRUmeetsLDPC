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
#include "crypto_kem_sntrup761.h" 
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
#define MAX_OUTPUT_SIZE 761  

#define OUTPUT_WEIGHT_FILENAME "/home/fei/C/SCA_Assisted_CCA_on_NTRU/NTRU_Prime/PC_Oracle_based_SCA/Attack_Simulations/sntrup761/SCA-LDPC/simulate-with-python/fprime_weight.txt"
#define OUTPUT_LDPC_PATH "/home/fei/C/SCA_Assisted_CCA_on_NTRU/NTRU_Prime/PC_Oracle_based_SCA/Attack_Simulations/sntrup761/SCA-LDPC/simulate-with-python/outfile.txt"

#define MAX_SECRET 9 // For secret values from -2 to 2
#define MAX_BIT_LENGTH 10

#define OUTPUT_FPRIME_PATH "/home/fei/C/SCA_Assisted_CCA_on_NTRU/NTRU_Prime/PC_Oracle_based_SCA/Attack_Simulations/sntrup761/SCA-LDPC/simulate-with-python/fprime_output.txt"


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
int *sec_index;  // 动态数组用指针定义


int weight_list_k1 = 200;
int weight_list_k2 = 300;

int collision_count;


void generate_sec_index_list(int sec_index_first_part[p][u_column_size]) {
    int index = 0;  // 用于追踪 sec_index_first_part 的存储位置

    for (int i = 0; i < p; i ++) {
        sec_index_first_part[index][0] = i;
        index++;
    }
}

#define table_row_size 5 
#define table_column_size 4  
#define new_table_row_size 9

// beta_u\in [-3,3], attack parameters for distingush 1
int c_value_for_attack_list_1[3][3] = 
{   
    {90,285,30}, //(45,45) 
    {96,279,30}, //(45,45) 
    {105,270,30}, //(45,45) 
};

//  attack parameters for distingush 2
int c_value_for_attack_list_2[3][3] = 
{   
    {84,267,42}, //(63,63) 
    {87,261,45}, //(63,63) 
    {99,252,42}, //(63,63) 
};


int param_list1_index;
int param_list2_index;
int param_which_list;

int column[table_row_size][1] = {0};
int column_2[4][1] = {0}; 

int actual_match_table[table_row_size][table_column_size] = {0}; // 定义并初始化 actual_match_table

int await_match_table[5][6] = {
    {0, 0, 0, 0, 0, -1}, //-2
    {0, 0, 0, 0, -1, -1}, //-1
    {0, 0, 0, -1, -1, -1},  //0
    {0, 0, -1, -1, -1, -1},  //1
    {0, -1, -1, -1, -1, -1}  //2
};

int await_match_table_2[2][3] = {
    {0, -1, -1}, 
    {0, 0, -1},
};

// Weight distribution of the private key
double average_match_table[table_row_size][1] = {
    {1.51027}, // -2
    {39.57822}, // -1
    {117.82300},  // 0
    {39.57822},  // 1
    {1.51027}  // 2
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

// Calculate distance function
double calculate_distance(double expected_zeros, double expected_minus_ones, double actual_zeros, double actual_minus_ones) {
    double distance = (expected_zeros - actual_zeros) * (expected_zeros - actual_zeros) +
                      (expected_minus_ones - actual_minus_ones) * (expected_minus_ones - actual_minus_ones);
    printf("Calculated distance: %f (expected_zeros: %f, actual_zeros: %f, expected_minus_ones: %f, actual_minus_ones: %f)\n",
           distance, expected_zeros, actual_zeros, expected_minus_ones, actual_minus_ones);
    return distance;
}


void update_match_table(double counts[1][2], double average_match_table[table_row_size][1]) {
    for (int col = 0; col < 1; col++) { // Only calculate the first column
        double min_distance = DBL_MAX;
        int best_y_index = 0;

        // Iterate through each possible y index
        for (int y_index = 3; y_index > -3; y_index--) 
        {
            double expected_zeros = 0.0;
            double expected_minus_ones = 0.0;

            // Convert y_index to the correct column index
            int column_index = 3 - y_index;

            // Calculate the expected number of 0s and -1s
            for (int row = 0; row < table_row_size; row++) {
                if (await_match_table[row][column_index] == 0) {
                    expected_zeros += average_match_table[row][0];
                    //  expected_zeros += average_match_table[row][0] + average_match_table[table_row_size-1-row][0];
                } else if (await_match_table[row][column_index] == -1) {
                    expected_minus_ones += average_match_table[row][0];
                    // expected_minus_ones += average_match_table[row][0] + average_match_table[table_row_size-1-row][0];
                }
            }
      
            // Calculate the actual number of 0s and -1s
            double actual_zeros = counts[col][0];
            double actual_minus_ones = counts[col][1];

            // Call calculate_distance function to compute distance
            double distance = calculate_distance(expected_zeros, expected_minus_ones, actual_zeros, actual_minus_ones);

            // Update the minimum distance and the best y index
            if (distance < min_distance) {
                min_distance = distance;
                best_y_index = y_index;

            }
        }      

        // Convert best_y_index to the correct column index
        int best_column_index = 3 - best_y_index;

        // Directly assign the await_match_table column (that satisfies the minimum distance) to match_table
        for (int row = 0; row < table_row_size; row++) {
            column[row][col] = await_match_table[row][best_column_index];
        }

       // Output the determined column values in column format
        printf("Determined column for col %d (column format):\n", col);
        for (int row = 0; row < table_row_size; row++) {
            printf("%d\n", column[row][col]);
        }
    }
}


void update_match_table_2(double counts[1][2], int new_array_size) {
    for (int col = 0; col < 1; col++) { // Only calculate the first column
        double min_distance = DBL_MAX;
        int best_y_index = 0;

        // Iterate through each possible y index (from 0 to 2)
        for (int y_index = 0; y_index < 3; y_index++) {
            double expected_zeros = 0.0;
            double expected_minus_ones = 0.0;

            // Compute expected_zeros and expected_minus_ones based on y_index
            if (y_index == 0) {
                // First column (y_index == 0): expected_zeros = column_count * (1 - 0), expected_minus_ones = column_count * 0
                expected_zeros = new_array_size * (1 - 0);
                expected_minus_ones = new_array_size * 0;
            } else if (y_index == 1) {
                // Second column (y_index == 1): expected_zeros = column_count * (1 - column_factor), expected_minus_ones = column_count * column_factor
                expected_zeros = new_array_size * (1 - 0.13083);
                expected_minus_ones = new_array_size * 0.13083;
            } else if (y_index == 2) {
                // Third column (y_index == 2): expected_zeros = column_count * 0, expected_minus_ones = column_count * 1
                expected_zeros = new_array_size * 0;
                expected_minus_ones = new_array_size * 1;
            }

            // Print expected_zeros and expected_minus_ones
            printf("For y_index = %d, expected_zeros = %f, expected_minus_ones = %f\n", y_index, expected_zeros, expected_minus_ones);

            // Calculate the actual number of 0s and -1s
            double actual_zeros = counts[col][0];
            double actual_minus_ones = counts[col][1];

            // Call calculate_distance function to compute distance
            double distance = calculate_distance(expected_zeros, expected_minus_ones, actual_zeros, actual_minus_ones);

            // Update minimum distance and best y index
            if (distance < min_distance) {
                min_distance = distance;
                best_y_index = y_index;
            }
        }

        // Select the best column index based on best_y_index
        int best_column_index = best_y_index;

        // Directly assign the await_match_table column (that satisfies the minimum distance) to match_table
        for (int row = 0; row < 2; row++) {
            column_2[row][col] = await_match_table_2[row][best_column_index];
        }

        // Output the determined column values in column format
        printf("Determined column for col %d (column format):\n", col);
        for (int row = 0; row < 2; row++) {
            printf("%d\n", column_2[row][col]);
        }
    }
}



#define max_x 3 

#include <stdio.h>

void calculate_beta_probabilities(long double *Pr, int beta_coeff_count) {
   
    for (int i = 0; i < 9; i++) {
        Pr[i] = 0.0L;
    }

    if (beta_coeff_count == 1) {
        Pr[3] = 143.0L / 761.0L;
        Pr[4] = 475.0L / 761.0L;
        Pr[5] = 143.0L / 761.0L;
    } else if (beta_coeff_count == 2) {
        Pr[2] = 20449.0L / 579121.0L;
        Pr[3] = 135850.0L / 579121.0L;
        Pr[4] = 266523.0L / 579121.0L;
        Pr[5] = 135850.0L / 579121.0L;
        Pr[6] = 20449.0L / 579121.0L;
    } else if (beta_coeff_count == 3) {
        Pr[1] = 2924207.0L   / 440711081.0L;
        Pr[2] = 29139825.0L  / 440711081.0L;
        Pr[3] = 105565746.0L / 440711081.0L;
        Pr[4] = 165451525.0L / 440711081.0L;
        Pr[5] = 105565746.0L / 440711081.0L;
        Pr[6] = 29139825.0L  / 440711081.0L;
        Pr[7] = 2924207.0L   / 440711081.0L;
    } else if (beta_coeff_count == 4) {
        Pr[0] = 418161601.0L     / 335381132641.0L; // z = -4
        Pr[1] = 5555993300.0L    / 335381132641.0L; // z = -3
        Pr[2] = 29355480154.0L   / 335381132641.0L; // z = -2
        Pr[3] = 77970292400.0L   / 335381132641.0L; // z = -1
        Pr[4] = 108781277731.0L  / 335381132641.0L; // z =  0
        Pr[5] = 77970292400.0L   / 335381132641.0L; // z =  1
        Pr[6] = 29355480154.0L   / 335381132641.0L; // z =  2
        Pr[7] = 5555993300.0L    / 335381132641.0L; // z =  3
        Pr[8] = 418161601.0L     / 335381132641.0L; // z =  4
    }
}


// Compute the base ciphertext

void generate_base_ciphertext(int *success_trial, unsigned char *ct, unsigned char *ss, unsigned char *pk, unsigned char *sk, int param_p, int *no_single_collisions, int *no_multiple_collisions, int *no_false_negative_collisions, int *no_false_positive_collisions, int kem_CIPHERTEXTBYTES, char *ct_file_now_basic, char *ct_file_basic_failed, int *collision_array_index, int *collision_array_value, int *collision_count,int *total_profile_trials_overall, int *false_positive_count, int *c_base_oracle_count, int c_base_vote_limit, int *overall_oracle_count) {
   
    int votes[p]; 
    while (*success_trial == 0) 
    {
        int got_minus_one = 0;
        int got_zero = 0;
        int count_non_zero_coeffs = 0; 
        *collision_count = 0; 
       
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
            (*c_base_oracle_count)++;  
            (*overall_oracle_count)++;

            
            for (int i = 0; i < param_p; i++) {
                if (er_decrypt[i] == 1 || er_decrypt[i] == -1) {
                    count_non_zero_coeffs++;
                }
            }

            // Introduce Bernoulli noise
            double noise_probability = 1.00;
            // double random_value = (double)rand() / RAND_MAX;

            unsigned char random_value_bytes[4];
            randombytes(random_value_bytes, sizeof(random_value_bytes));  // Get a 4-byte random number

            // Convert the 4-byte random number into a 32-bit unsigned integer
            uint32_t random_value_int = (random_value_bytes[0] | (random_value_bytes[1] << 8) | (random_value_bytes[2] << 16) | (random_value_bytes[3] << 24));
            
            // Normalize the 32-bit integer to the [0, 1] interval to ensure the random value is between [0, 1]
            double random_value = (double)random_value_int / (double)UINT32_MAX;


            if (random_value >= noise_probability) {
                // int noise_type = rand() % 2;

                // Use randombytes to generate a new random byte sequence instead of rand()
                unsigned char noise_type_bytes[1];
                randombytes(noise_type_bytes, sizeof(noise_type_bytes));  // Get a 1-byte random number

                // Convert the 1-byte random number into 0 or 1
                int noise_type = noise_type_bytes[0] % 2;
                if (noise_type == 0) {
                    count_non_zero_coeffs++;
                } else if (count_non_zero_coeffs > 0) {
                    count_non_zero_coeffs--;
                }
            }

            // Add count_non_zero_coeffs into the voting array
            votes[vote] = count_non_zero_coeffs;
            printf("For vote is %d; random_value is %f; count_non_zero_coeffs is %d\n", vote, random_value, count_non_zero_coeffs);
        }

        // Majority voting result updates count_non_zero_coeffs
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
        // Use count_non_zero_coeffs for subsequent logic decisions
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

// Construct the base ciphertext, and get the oracle output

void get_oracle_response(unsigned char *ct, unsigned char *ss, unsigned char *ss1, unsigned char *pk, unsigned char *sk,
                         int sec_index_first_part[u_row_list_size][u_column_size], int *get_er_decrypt_array_all, int check1_array[], int param_list1_index, 
                         int param_list2_index, int start_idx, int end_idx, int vote_limit, int *overall_oracle_count) 
{
    int get_er_decrypt_array[1];
    int count_get_er_decrypt_array_all = 0;
    int array_size = end_idx - start_idx;

    // Iterate over the index range
    for (int idx1 = start_idx; idx1 < end_idx; idx1++) 
    {
        
        for (int i = 0; i < u_column_size; i++) 
        {           
            sec_index[i] = sec_index_first_part[idx1][i];
            // printf("In test.c sec_index[%d] is %d\n", i, sec_index[i]); 
           
        }

        // Determine the value of mul_value based on check1_array
        int check1 = check1_array[idx1];
        mul_value = (check1 == 0) ? 1 : -1;

        // Voting logic: multiple queries and result counting
        int vote_count_0 = 0;    // Count how many times the result is 0
        int vote_count_neg1 = 0; // Count how many times the result is -1

        for (int vote = 0; vote < vote_limit; vote++) 
        {
            // Encryption and decryption
            crypto_kem_enc(ct, ss, pk);
            crypto_kem_dec(ss1, ct, sk);

            (*overall_oracle_count)++;
          
            // Compute the weight of er_decrypt
            weight_hh = 0;
            for (int jh = 0; jh < p; jh++) 
            {
                if (abs(er_decrypt[jh]) > 0) 
                {
                    weight_hh++;
                }
            }

            // Compute the current result based on weight
            int current_result = (weight_hh != 0) ? -1 : 0;

            // Use randombytes() to generate random numbers to introduce Bernoulli noise
            unsigned char random_value_bytes[4];
            randombytes(random_value_bytes, sizeof(random_value_bytes));  // Get a 4-byte random number

            // Convert the 4-byte random number into a 32-bit unsigned integer
            uint32_t random_value_int = (random_value_bytes[0] | (random_value_bytes[1] << 8) | (random_value_bytes[2] << 16) | (random_value_bytes[3] << 24));
            
            // Normalize the 32-bit integer to the [0, 1] interval to ensure the random value is within [0, 1]
            double random_value = (double)random_value_int / (double)UINT32_MAX;

            // Flip the current result depending on the random value and noise probability
            double noise_probability = 1.00;
            if (random_value >= noise_probability) 
            {
                current_result = (current_result == -1) ? 0 : -1;

            }
          
            // Count the current result
            if (current_result == 0) {
                vote_count_0++;
            } else if (current_result == -1) {
                vote_count_neg1++;
            }
        }

        // Set the final value based on the voting result
        get_er_decrypt_array[0] = (vote_count_0 >= vote_count_neg1) ? 0 : -1;
        // Store the final value into get_er_decrypt_array_all
        if (count_get_er_decrypt_array_all < array_size) {
            get_er_decrypt_array_all[count_get_er_decrypt_array_all] = get_er_decrypt_array[0];
            count_get_er_decrypt_array_all++;
        } else {
            printf("Error: get_er_decrypt_array_all index out of bounds.\n");
        }
    }
}


// Determine if the obtained distinguisher matches our expectation

bool check_distinguisher(int *current_state, int match_table[table_row_size][1], 
           int await_value[table_row_size][6], int *current_dis, 
           int column_2[4][1], int await_match_table_2[2][3], 
           int *current_tree_state) 
{
    bool state_updated = false;

    switch (*current_state) 
    {
        case 0:
            if (match_table[0][0] == await_value[0][2] && 
                match_table[1][0] == await_value[1][2] &&
                match_table[2][0] == await_value[2][2] && 
                match_table[3][0] == await_value[3][2] &&
                match_table[4][0] == await_value[4][2]) 
            {
                state_updated = true;
                *current_state = 0;
                *current_dis = 1;
                *current_tree_state = 1;
            }
            else if (match_table[0][0] == await_value[0][3] && 
                     match_table[1][0] == await_value[1][3] &&
                     match_table[2][0] == await_value[2][3] && 
                     match_table[3][0] == await_value[3][3] &&
                     match_table[4][0] == await_value[4][3]) 
            {
                state_updated = true;
                *current_state = 0;
                *current_dis = 1;
                *current_tree_state = 2;
            }
            
            else if (match_table[0][0] == await_value[0][1] && 
                     match_table[1][0] == await_value[1][1] &&
                     match_table[2][0] == await_value[2][1] && 
                     match_table[3][0] == await_value[3][1] &&
                     match_table[4][0] == await_value[4][1]) 
            {
                state_updated = false;
                *current_state = 0;
                *current_dis = 1;
                *current_tree_state = 3;
            } 
            
            else if (match_table[0][0] == await_value[0][4] && 
                     match_table[1][0] == await_value[1][4] &&
                     match_table[2][0] == await_value[2][4] && 
                     match_table[3][0] == await_value[3][4] &&
                     match_table[4][0] == await_value[4][4]) 
            {
                state_updated = false;
                *current_state = 0;
                *current_dis = 1;
                *current_tree_state = 4;
            }
            
            else if ((match_table[0][0] == await_value[0][0] && 
                      match_table[1][0] == await_value[1][0] &&
                      match_table[2][0] == await_value[2][0] && 
                      match_table[3][0] == await_value[3][0] &&
                      match_table[4][0] == await_value[4][0]) ||
                     (match_table[0][0] == await_value[0][5] && 
                      match_table[1][0] == await_value[1][5] &&
                      match_table[2][0] == await_value[2][5] && 
                      match_table[3][0] == await_value[3][5] &&
                      match_table[4][0] == await_value[4][5])) 
            {
                state_updated = false;
                *current_state = 0;
                *current_dis = 1;
                *current_tree_state = 4;
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
   
    return state_updated;
}



// Node structure for the tree
typedef struct Node {
    bool ge_flag;       // A boolean flag indicating whether to perform a '>= value' comparison
    int value;          // The integer value stored in the node
    struct Node *left;  // Pointer to the left child node
    struct Node *right; // Pointer to the right child node
} Node;

// SimpleOracle structure
typedef struct {
    double p_prob;      // The accuracy probability of the Oracle, ranging from 0 to 1
    int oracle_calls;   // Records the number of times the Oracle has been called
} SimpleOracle;

// False Positive/Negative Oracle structure
typedef struct {
    double **p_positional; // A 2D array representing the false positive and false negative probabilities for each position
    int oracle_calls;      // Records the number of times the Oracle has been called
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
    
    // Print the random number and the oracle accuracy (p_prob)
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
        
        // Set b: -1 represents the right subtree, 0 represents the left subtree
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
    // Allocate memory for the FalsePositiveNegativePositionalOracle
    FalsePositiveNegativePositionalOracle *oracle = (FalsePositiveNegativePositionalOracle *)malloc(sizeof(FalsePositiveNegativePositionalOracle));

    // Initialize the struct members
    oracle->p_positional = p_positional;  // Use a long double type array
    oracle->oracle_calls = 0;

    return oracle;
}


// Modified: Find value in secret distribution
long double find_in_secret_distrib(long double *Pr, int key, int sum_weight) {
    int index = key + sum_weight;

    // Check if the index is within range
    if (index >= 0 && index < MAX_SECRET) {
        return Pr[index];  // Return the probability value from Pr
    } else {
        return 0.0L; // Return 0.0L if the index is invalid
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
    
    // Iterate over the range of sum_weight
    for (int i = -sum_weight; i <= sum_weight; i++) {
        long double pr_x_y = find_in_secret_distrib(Pr, i, sum_weight) * 
                             pr_cond_yx_adaptive(y, y_len, i, pr_oracle, coding_tree, y_prob);  // Pass y_prob
        pr_y += pr_x_y;
    }

    // Return Pr[X = e(s) | Y = y]
    return pr_cond_yx_adaptive(y, y_len, s, pr_oracle, coding_tree, y_prob) * 
           find_in_secret_distrib(Pr, s, sum_weight) / pr_y;
}


// Initialize oracle accuracy data with 1 positions
long double **init_oracle_accuracy() {
    long double **oracle_accuracy = (long double **)malloc(1 * sizeof(long double *));
    
    oracle_accuracy[0] = (long double *)malloc(2 * sizeof(long double));
    // oracle_accuracy[0][0] = 0.001224L;  // False positive probability
    oracle_accuracy[0][0] = 0.005000L;  // False positive probability
    oracle_accuracy[0][1] = 0.005000L;  // False negative probability


    return oracle_accuracy;
}



int intended_function;
extern int collision_index;
extern int collision_value;
extern int error_now;
extern int hw_value;
extern int m;
extern int n;
extern int mul_value;
int c3_value_1, c3_value_2, c3_value_3;
extern int *c_value_for_attack_1_1;
extern int *c_value_for_attack_1_2;
extern int *c_value_for_attack_1_3;
extern int pq_counter;
extern int c1_value_1, c1_value_2, c1_value_3;
extern int c2_value_1, c2_value_2, c2_value_3;

unsigned char entropy_input[48];
unsigned char seed[NO_TESTS][48];

int zeros;
int vote_limit;
int c_base_vote_limit; 
int cond_prob_loop;
int state_oracle_now;
int current_state;
int theoretical_loop_count;
int overall_oracle_count;

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
   

    int param_p = 761; 
    
    // Initialize the random seed, only needs to be called once
    // srand(time(NULL));

    for (int i = 0; i < 48; i++) 
    {
        entropy_input[i] = get_random() & 0xFF;        
    }

    // Initialize the pseudo-random number generator once
    randombytes_init(entropy_input, NULL, 256);

    // Generate seeds
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
    int c_value_current;
    //int final_coeff_to_find;

    FILE * f2;
    FILE * f3;

    #if (DO_PRINT == 1)

    char ct_file_now[30];
    char ct_file_now_basic[30];
    char keypair_file[30];
    char ct_file_basic_failed[50];

    // We can store the data of a single iteration in files...
    // Please note that these files will be overwritten for every iteration...
    // Here, we store the attack ciphertexts...
    sprintf(ct_file_now,"ct_file_now.bin");

    // Here, we store the base ciphertext...
    sprintf(ct_file_now_basic,"ct_file_basic.bin");

    // Here, we store the public and private key pair...
    sprintf(keypair_file,"keypair_file.bin");

    // Here, we store the failed ciphertexts which do not correspond to any collision...
    sprintf(ct_file_basic_failed,"ct_file_basic_failed.bin");
  
    
    // Open and clear the file to store the private key and collision information
    FILE *private_key_file = fopen("private_key_and_collision_info.bin", "w");
    if (private_key_file == NULL) {
        printf("Failed to open file: private_key_and_collision_info.bin\n");
        return 1;
    }

    // Open a new file for storing the actual_match_table
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


    // Open and clear "parameters_list_for_attack_ciphertext"
    FILE *parameters_list_for_attack_ciphertext_file = fopen("parameters_list_for_attack_ciphertext.bin", "w");
    if (parameters_list_for_attack_ciphertext_file == NULL) {
        printf("Failed to open file: parameters_list_for_attack_ciphertext.bin\n");
        return 1;
    }
    fclose(parameters_list_for_attack_ciphertext_file);
 
    
    // Open and clear "actual_oracle_response.txt"
    FILE *actual_oracle_response_file = fopen("actual_oracle_response.txt", "w");
    if (actual_oracle_response_file == NULL)
    {
        printf("Failed to open file: actual_oracle_response.txt\n");
        return 1;
    }
    fclose(actual_oracle_response_file);


    FILE *ldpc_final_output_file = fopen("ldpc_final_output.txt", "w");
    if (ldpc_final_output_file == NULL)
    {
        printf("Failed to open file: ldpc_final_output.txt\n");
        return 1;
    }
    fclose(ldpc_final_output_file);  

    FILE *ldpc_fprime_output_file = fopen("ldpc_fprime_output.txt", "w");
    if (ldpc_final_output_file == NULL)
    {
        printf("Failed to open file: ldpc_fprime_output.txt\n");
        return 1;
    }
    fclose(ldpc_fprime_output_file);  

    FILE *secret_key_weight_file = fopen("secret_key_weight.txt", "w");
    if (secret_key_weight_file == NULL)
    {
        printf("Failed to open file: secret_key_weight.txt\n");
        return 1;
    }
    fclose(secret_key_weight_file);  
    
    // 清空 d1_d2_data.txt 文件内容
    FILE *d1_d2_data_file = fopen("d1_d2_data.txt", "w");
    if (d1_d2_data_file == NULL) {
        perror("Failed to open file: d1_d2_data\n");
        return 1;
    }
    fclose(d1_d2_data_file);

    // 打开并清空文件 "Oracle_calls_count"
    FILE *overall_oracle_calls_count_file = fopen("Overall_oracle_calls_count.txt", "w");
    if (overall_oracle_calls_count_file == NULL)
    {
        printf("Failed to open file: Overall_oracle_calls_count.txt\n");
        return 1;
    }
    fclose(overall_oracle_calls_count_file);

    #endif
   

    
    m = M_VALUE;
    n = N_VALUE;

    // This is used to calculate t1 and t2 for the base ciphertext cbase, as described in the paper...

    int max_distance = 1000000;
    int max_distance2 = 0;
    int max_min_distance = 0;
    while(found_c == 0)
    {
      for(int hg = 0; hg < q; hg++)
      {

        for(int hg1 = 0; hg1 < q; hg1++)
        {

              c_value_current = hg;
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

                        if((value1 < q12) || (value2 > q12)) 
                        // || (hg%3 != 0) || (hg1%3 != 0) || (abs(q12 - value1) < C_VALUE_THRESHOLD_1) || (abs(q12 - value2) < C_VALUE_THRESHOLD_2))
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
            if(max_distance >= 45)
            {
              float weight_score = 1.0 *abs(q12 - value1) + 1.0 * max_distance;

                if(touch_np == 0)
                {
 
                    if (dist_value == 1 && weight_score >= 80 && abs(q12 - value1) >= max_distance)
                    {
                        // For dist_value = 1, output all combinations with weight_score >= 120
                        c_value_for_attack_1_1[dist_value - 1] = hg;
                        c_value_for_attack_1_2[dist_value - 1] = hg1;
                        c_value_for_attack_1_3[dist_value - 1] = hg2;
                        max_distance2 = max_distance;
                        max_weight_score = weight_score;
                        printf("hg = %d, hg1 = %d, hg2 = %d, Diff1: %d, Diff2: %d, weight_score: %f\n", hg, hg1, hg2, abs(q12 - value1), max_distance2, weight_score);
                    }
                    else if (dist_value == 2 && weight_score >= 100 && abs(q12 - value1) >= max_distance)
                    {
                        // For dist_value = 2, output all combinations with weight_score >= 210
                        c_value_for_attack_1_1[dist_value - 1] = hg;
                        c_value_for_attack_1_2[dist_value - 1] = hg1;
                        c_value_for_attack_1_3[dist_value - 1] = hg2;
                        max_distance2 = max_distance;
                        max_weight_score = weight_score;
                        printf("hg = %d, hg1 = %d, hg2 = %d, Diff1: %d, Diff2: %d, weight_score: %f\n", hg, hg1, hg2, abs(q12 - value1), max_distance2, weight_score);
                    }
                    else if (dist_value == 3 && weight_score >= 200 && abs(q12 - value1) >= max_distance)
                    {
                        // For dist_value = 3, output all combinations with weight_score >= 210
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
    int rounding_error_return = 0;
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
        int c_base_oracle_count = 0;
        
        int c_base_trying=0;
        overall_oracle_count = 0;
       
      
    //    while (successful_attack_done == 0)
    //    {
                // Step 1
                rej:
                printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%TRIAL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
                
                // Generate the index list U1
                u_column_size = 1; 
                sec_index = (int *)malloc(u_column_size * sizeof(int));
                if (sec_index == NULL) {
                    printf("Memory allocation failed for sec_index.\n");
                    return 1;
                }

                int sec_index_first_part[u_row_list_size][1] = {0};
                generate_sec_index_list(sec_index_first_part);
                
                printf("the first part is ...\n");
                for (int i = 0; i < u_row_list_size; i++) {
                    printf("{%d}\n", sec_index_first_part[i][0]);
                }
                
                // Try to find a collision...
                collision_count = 0;  

                int collision_array_index[param_p]; 
                int collision_array_value[param_p]; 
                int false_positive_count = 0;  
            
                for (int i = 0; i < param_p; i++) {
                    collision_array_index[i] = 0;
                    collision_array_value[i] = 0;
                }

                intended_function = 0;
                c_base_vote_limit = 1;
                
                int success_trial = 0;
                generate_base_ciphertext(&success_trial, ct, ss, pk, sk, param_p, &no_single_collisions, &no_multiple_collisions, &no_false_negative_collisions, &no_false_positive_collisions, crypto_kem_CIPHERTEXTBYTES, ct_file_now_basic, ct_file_basic_failed, collision_array_index, collision_array_value, &collision_count,&total_profile_trials_overall, &false_positive_count, &c_base_oracle_count, c_base_vote_limit, &overall_oracle_count);
                c_base_trying++;
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

                // Write collision information and private key data for debug checking
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
                int initial_capacity = 50; // Fixed capacity, no need for realloc
                Node **coding_arr_1 = (Node **)malloc(initial_capacity * sizeof(Node *));

                if (coding_arr_1 == NULL) {
                    printf("Memory allocation failed for first tree\n");
                    return 0;
                }

                collision_index = collision_array_index[0];
                collision_value = collision_array_value[0];

                // oracle_response_1 is used to store the oracle response
                int oracle_response_1_capacity = u_row_list_size; 
                int **oracle_response_1 = NULL; 
                int oracle_response_1_size = 0; 
                int *oracle_response_1_col_sizes = NULL; 

                // Free oracle_response_1
                if (oracle_response_1 != NULL) {
                    for (int i = 0; i < oracle_response_1_capacity; i++) {
                        free(oracle_response_1[i]); 
                    }
                    free(oracle_response_1); 
                    oracle_response_1 = NULL; 
                }

                if (oracle_response_1_col_sizes != NULL) {
                    free(oracle_response_1_col_sizes); 
                    oracle_response_1_col_sizes = NULL; 
                }                

                // Dynamically allocate initial memory for oracle_response_1, stored by columns
                oracle_response_1 = (int **)malloc(oracle_response_1_capacity * sizeof(int *));
                oracle_response_1_col_sizes = (int *)malloc(oracle_response_1_capacity * sizeof(int)); // 记录每列的长度
                

                if (oracle_response_1 == NULL || oracle_response_1_col_sizes == NULL) {
                    printf("Memory allocation failed for oracle_response_1 \n");
                    return 1;
                }
                
                // Initialize the length of each column to 0, as no data has been filled yet
                for (int i = 0; i < oracle_response_1_capacity; i++) {
                    oracle_response_1[i] = NULL;  
                    oracle_response_1_col_sizes[i] = 0; 
                }
               
                // Used to store information of p_pos and p_neg
                int Pos_neg_index_first_capacity = u_row_list_size; 
                int **Pos_neg_index_first = NULL;
                int Pos_neg_index_first_size = 0; 
                int *Pos_neg_index_first_col_sizes = NULL; 

                if (Pos_neg_index_first != NULL) {
                    for (int i = 0; i < Pos_neg_index_first_capacity; i++) {
                        free(Pos_neg_index_first[i]); 
                    }
                    free(Pos_neg_index_first); 
                    Pos_neg_index_first = NULL;
                }
                if (Pos_neg_index_first_col_sizes != NULL) {
                    free(Pos_neg_index_first_col_sizes); 
                    Pos_neg_index_first_col_sizes = NULL;
                }
               
                Pos_neg_index_first = (int **)malloc(Pos_neg_index_first_capacity * sizeof(int *));
                Pos_neg_index_first_col_sizes = (int *)malloc(Pos_neg_index_first_capacity * sizeof(int)); // 记录每列的长度

                if (Pos_neg_index_first == NULL || Pos_neg_index_first_col_sizes == NULL) {
                    printf("Memory allocation failed for Pos_neg_index_first or Pos_neg_index_first_col_sizes\n");
                    return 1;
                }

                for (int i = 0; i < Pos_neg_index_first_capacity; i++) {
                    Pos_neg_index_first[i] = NULL;  
                    Pos_neg_index_first_col_sizes[i] = 0;
                }    

                // Used to store the adopted ciphertext parameters for the attack
                ParamRecord *param_records = NULL;  
                int param_record_count = 0;         
                int param_record_capacity = 20;     

                param_records = (ParamRecord *)malloc(param_record_capacity * sizeof(ParamRecord));
                if (param_records == NULL) {
                    printf("Memory allocation failed for param_records\n");
                    return 1;
                }

                ParamRecord *paird_param_records = NULL;  
                int paird_param_record_count = 0;         
                int paird_param_record_capacity = 20;     

                paird_param_records = (ParamRecord *)malloc(paird_param_record_capacity * sizeof(ParamRecord));
                if (paird_param_records == NULL) {
                    printf("Memory allocation failed for paird_param_records\n");
                    return 1;
                }

                int *check1_array = NULL;  
                
                int *updated_sec_index_first_part = NULL;  
                int *updated_check1_array = NULL;  
                int *updated_index_array = NULL;  
                int updated_array_size = 0;  
                int updated_index_array_size = 0;  
             
                int *adi_sec_index_first_part = NULL;   
                int adi_array_size = 0;                           
                int *adi_check1_array = NULL;           
                int *adi_index_array_first = NULL;      
                int adi_index_array_size = 0;           
              
                int *new_sec_index_first_part = NULL;   
                int new_array_size = 0;                          
                int *new_check1_array = NULL;    
                int *index_array_first = NULL;
                int index_array_size = 0;
                
         
 
                int *get_er_decrypt_array_all = NULL;
                               
                int start_idx;
                int end_idx;
                int check1 = 0;
                int current_position = 0;  
                int array_size;               
                current_state = 0;
                int current_dis = 0;
                int current_tree_state = 0;                                               
                intended_function = 1;
                param_list1_index = 0;  
                param_list2_index = 0;  
                param_which_list=0;
                vote_limit = 1;                
                int param_value_to_fill = -1;
                cond_prob_loop =0;                                
                bool state_updated = false;  


                // Create attack ciphertext and collect oracle responses
                while (cond_prob_loop < 1)
                {
                    cond_rej:
                    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%cond_prob_TRIAL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");

                    u_column_size = 1; 
                    state_oracle_now = 0;                   
                    current_state = 0; 
                    tree_size_1=0;                    
                    
                    int backup_sec_index_first_part[u_row_list_size][u_column_size];
                    memcpy(backup_sec_index_first_part, sec_index_first_part, u_row_list_size * sizeof(*sec_index_first_part));
                    bool use_new_sec_index = false;                
                    state_updated = false; 

                    while (true)
                    {  
                        // use_new_sec_index indicates whether to perform evaluation on ds2 
                        if (use_new_sec_index) 
                        {
                            
                            memcpy(sec_index_first_part, new_sec_index_first_part, new_array_size * sizeof(*new_sec_index_first_part));
                            start_idx = 0;
                            check1_array = new_check1_array; 
                            end_idx = new_array_size;
                            array_size = new_array_size;      
                            vote_limit = 1;
                           
 
                            free(get_er_decrypt_array_all);
                            get_er_decrypt_array_all = (int *)malloc(array_size * sizeof(int));
                            if (get_er_decrypt_array_all == NULL) {
                                printf("Memory allocation failed\n");
                                return 1;
                            }                      
                            get_oracle_response(ct, ss, ss1, pk, sk,
                                                sec_index_first_part, 
                                                get_er_decrypt_array_all, check1_array, 
                                                param_list1_index, param_list2_index, 
                                                start_idx, end_idx, 
                                                vote_limit, &overall_oracle_count);                           
                            printf("cond_prob_loop is %d. for new sec_index Actual oracle response is...\n", cond_prob_loop);
                            for (int i = 0; i < array_size; i++) {
                                printf("%d,", get_er_decrypt_array_all[i]);
                            }
                            printf("\n");      
                            double counts[1][2];                 
                            calculate_counts(get_er_decrypt_array_all, array_size, counts);
                        
                            update_match_table_2(counts, new_array_size);
                        
                            printf("current_state is %d\n", current_state);
                            // Call the check_distinguisher function to determine which column matches
                            state_updated = check_distinguisher(&current_state, column, await_match_table, &current_dis, column_2, await_match_table_2, &current_tree_state);                            
                           
                        } 
                        else if (cond_prob_loop == 0) 
                        {

                            // Now compute the actual oracle response for (l_11, l_12, l_13) with u1 in [0, 200]; the check_distinguisher function needs to be used here
                            start_idx = 0;
                            end_idx = weight_list_k1;
                            array_size = end_idx - start_idx;
                            vote_limit = 1;
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

                          
                             get_oracle_response(ct, ss, ss1, pk, sk,
                                                sec_index_first_part, 
                                                get_er_decrypt_array_all, check1_array, 
                                                param_list1_index, param_list2_index, 
                                                start_idx, end_idx, 
                                                vote_limit, &overall_oracle_count);
                            printf("her1 Actual oracle response is...\n");
                            for (int i = 0; i < array_size; i++) {
                                printf("%d,", get_er_decrypt_array_all[i]);
                            }
                            printf("\n");
                                                    
                            // According to actual oracle response, compute column...
                            int size = array_size;
                            double counts[1][2];
                            
                            calculate_counts(get_er_decrypt_array_all, size, counts);
                            update_match_table(counts, average_match_table);
                                                    
                            // Call the check_distinguisher function to determine which column matches the current column
                            state_updated = check_distinguisher(&current_state, column, await_match_table, &current_dis, column_2, await_match_table_2, &current_tree_state);                                                                                
                                    
                        }
                      
                        printf("State updated: %d\n", state_updated);
                        printf("Current state: %d\n", current_state);
                        printf("Current dis: %d\n", current_dis); 

                        // If the expected distinguisher ds0/ds1 is obtained; or ds2 is obtained when use_new_sec_index is true                        
                        
                        if (state_updated) 
                        {                              
                            // For checking purposes only, save the corresponding parameters used to construct the attack ciphertext
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
                                // If ds0 or ds1 is obtained, 
                                // Store the obtained oracle response in the array oracle_response_1,
                                // and store the corresponding p_pos and p_neg of this ciphertext into Pos_neg_index_first

                                if(cond_prob_loop == 0)
                                {
                                    param_value_to_fill = 0;  

                                    for (int idx1 = 0; idx1 < array_size; idx1++) 
                                    {
                                        if (oracle_response_1[idx1] == NULL) {

                                            oracle_response_1[idx1] = (int *)malloc(1 * sizeof(int)); 
                                            oracle_response_1_col_sizes[idx1] = 1; 
                                            if (oracle_response_1[idx1] == NULL) {
                                                printf("Memory allocation failed for oracle_response_1[%d]\n", idx1);
                                                return 1;
                                            }
                                        } else {
                                            int current_col_size = oracle_response_1_col_sizes[idx1];
                                            oracle_response_1[idx1] = (int *)realloc(oracle_response_1[idx1], (current_col_size + 1) * sizeof(int));
                                            if (oracle_response_1[idx1] == NULL) {
                                                printf("Memory reallocation failed for oracle_response_1[%d]\n", idx1);
                                                return 1;
                                            }
                                            oracle_response_1_col_sizes[idx1] = current_col_size + 1;
                                        }

                                        int current_col_size = oracle_response_1_col_sizes[idx1];
                                        oracle_response_1[idx1][current_col_size - 1] = get_er_decrypt_array_all[idx1]; 

                                        if (Pos_neg_index_first[idx1] == NULL) {
                                            Pos_neg_index_first[idx1] = (int *)malloc(1 * sizeof(int));  
                                            Pos_neg_index_first_col_sizes[idx1] = 1;  
                                            if (Pos_neg_index_first[idx1] == NULL) {
                                                printf("Memory allocation failed for Pos_neg_index_first[%d]\n", idx1);
                                                return 1;
                                            }
                                        } else {
                                            int current_col_size = Pos_neg_index_first_col_sizes[idx1];
                                            Pos_neg_index_first[idx1] = (int *)realloc(Pos_neg_index_first[idx1], (current_col_size + 1) * sizeof(int));
                                            if (Pos_neg_index_first[idx1] == NULL) {
                                                printf("Memory reallocation failed for Pos_neg_index_first[%d]\n", idx1);
                                                return 1;
                                            }
                                            Pos_neg_index_first_col_sizes[idx1] = current_col_size + 1;
                                        }

                                        current_col_size = Pos_neg_index_first_col_sizes[idx1];
                                        Pos_neg_index_first[idx1][current_col_size - 1] = param_value_to_fill;
                                    }

                                    printf("Contents of oracle_response_1 (Column-wise):\n");
                                    for (int col = 0; col < oracle_response_1_capacity; col++) {
                                        printf("Column %d: ", col);
                                        for (int row = 0; row < oracle_response_1_col_sizes[col]; row++) {
                                            printf("%d ", oracle_response_1[col][row]);
                                        }
                                        printf("\n");
                                    }

                                    printf("Contents of Pos_neg_index_first (Column-wise):\n");
                                    for (int col = 0; col < Pos_neg_index_first_capacity; col++) {
                                        printf("Column %d: ", col);
                                        for (int row = 0; row < Pos_neg_index_first_col_sizes[col]; row++) {
                                            printf("%d ", Pos_neg_index_first[col][row]);
                                        }
                                        printf("\n");
                                    }

                                    Pos_neg_index_first_size++;
                                    oracle_response_1_size++;
                                    // Now compute the actual oracle response for (l_11, l_12, l_13) with u in [201, 300]
                                    // Store the obtained results in the corresponding positions of the first column of oracle_response_1

                                    start_idx = weight_list_k1;
                                    end_idx = weight_list_k2;
                                    array_size = end_idx - start_idx;
                                    vote_limit=1;
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
                                                        sec_index_first_part, 
                                                        get_er_decrypt_array_all, check1_array, 
                                                        param_list1_index, param_list2_index, 
                                                        start_idx, end_idx, 
                                                        vote_limit, &overall_oracle_count);
                                    
                                    printf("For new sec_index, actual oracle response is...\n");
                                    for (int i = 0; i < array_size; i++) {
                                        printf("%d,", get_er_decrypt_array_all[i]);
                                    }
                                    printf("\n");                            
                                    // Store the obtained oracle response in the array oracle_response_1,
                                    // and store the corresponding p_pos and p_neg of this ciphertext into Pos_neg_index_first
                                    for (int idx1 = start_idx; idx1 < end_idx; idx1++) 
                                    {
                                        if (oracle_response_1[idx1] == NULL) {
                                            oracle_response_1[idx1] = (int *)malloc(1 * sizeof(int));
                                            oracle_response_1_col_sizes[idx1] = 1;
                                            if (oracle_response_1[idx1] == NULL) {
                                                printf("Memory allocation failed for oracle_response_1[%d]\n", idx1);
                                                return 1;
                                            }
                                        } else {
                                            int current_col_size = oracle_response_1_col_sizes[idx1];
                                            oracle_response_1[idx1] = (int *)realloc(oracle_response_1[idx1], current_col_size * sizeof(int));
                                            if (oracle_response_1[idx1] == NULL) {
                                                printf("Memory reallocation failed for oracle_response_1[%d]\n", idx1);
                                                return 1;
                                            }
                                        }

                                        int current_col_size = oracle_response_1_col_sizes[idx1];
                                        oracle_response_1[idx1][current_col_size - 1] = get_er_decrypt_array_all[idx1 - start_idx];

                                        if (Pos_neg_index_first[idx1] == NULL) {
                                            Pos_neg_index_first[idx1] = (int *)malloc(1 * sizeof(int));
                                            Pos_neg_index_first_col_sizes[idx1] = 1;
                                            if (Pos_neg_index_first[idx1] == NULL) {
                                                printf("Memory allocation failed for Pos_neg_index_first[%d]\n", idx1);
                                                return 1;
                                            }
                                        } else {
                                            int current_col_size = Pos_neg_index_first_col_sizes[idx1];
                                            Pos_neg_index_first[idx1] = (int *)realloc(Pos_neg_index_first[idx1], current_col_size * sizeof(int));
                                            if (Pos_neg_index_first[idx1] == NULL) {
                                                printf("Memory reallocation failed for Pos_neg_index_first[%d]\n", idx1);
                                                return 1;
                                            }
                                        }
                                        current_col_size = Pos_neg_index_first_col_sizes[idx1];
                                        Pos_neg_index_first[idx1][current_col_size - 1] = param_value_to_fill;
                                    }

                                    oracle_response_1_size++;
                                    printf("Contents of oracle_response_1 (Column-wise): oracle_response_1_size is %d\n", oracle_response_1_size);
                                    for (int col = 0; col < oracle_response_1_capacity; col++) {
                                        printf("Column %d: ", col);
                                        for (int row = 0; row < oracle_response_1_col_sizes[col]; row++) {
                                            printf("%d ", oracle_response_1[col][row]);
                                        }
                                        printf("\n");
                                    }

                                    printf("Contents of Pos_neg_index_first (Column-wise):\n");
                                    for (int col = 0; col < Pos_neg_index_first_capacity; col++) {
                                        printf("Column %d: ", col);
                                        for (int row = 0; row < Pos_neg_index_first_col_sizes[col]; row++) {
                                            printf("%d ", Pos_neg_index_first[col][row]);
                                        }
                                        printf("\n");
                                    }

                                    // Determine the next action based on the current oracle response value
                                    // current_tree_state == 1 corresponds to ds1, current_tree_state == 2 corresponds to ds0
                                    for (int col = 0; col < oracle_response_1_capacity; col++)  // 遍历每一列
                                    {
                                        if (oracle_response_1_col_sizes[col] > 0)
                                        {
                                            int val_0 = oracle_response_1[col][0];  
                                            if((current_tree_state == 1 && val_0 == 0)|| (current_tree_state == 2 && val_0 == -1))
                                            {
                                                updated_sec_index_first_part = (int *)realloc(updated_sec_index_first_part, (updated_array_size + 1) * sizeof(int));
                                                updated_check1_array = (int *)realloc(updated_check1_array, (updated_array_size + 1) * sizeof(int));
                                                updated_index_array = (int *)realloc(updated_index_array, (updated_index_array_size + 1) * sizeof(int));  
                                                if (updated_sec_index_first_part == NULL || updated_check1_array == NULL || updated_index_array == NULL) {
                                                    printf("Memory allocation failed during resizing.\n");
                                                    free(updated_sec_index_first_part);
                                                    free(updated_check1_array);
                                                    free(updated_index_array);
                                                    return 1;
                                                }

                                                updated_sec_index_first_part[updated_array_size] = sec_index_first_part[col][0];  
                                                updated_check1_array[updated_array_size] = 1;
                                                updated_index_array[updated_index_array_size] = col;
                                                updated_index_array_size++;
                                                updated_array_size++;                                          
                                            
                                            }                                        
                                            else if ((current_tree_state == 1 && val_0 == -1) || (current_tree_state == 2 && val_0 == 0)) 
                                            {
                                                new_sec_index_first_part = (int *)realloc(new_sec_index_first_part, (new_array_size + 1) * sizeof(int));
                                                new_check1_array = (int *)realloc(new_check1_array, (new_array_size + 1) * sizeof(int));
                                                index_array_first = (int *)realloc(index_array_first, (index_array_size + 1) * sizeof(int)); 
                                                if (new_sec_index_first_part == NULL || new_check1_array == NULL || index_array_first == NULL) {
                                                    printf("Memory allocation failed during resizing.\n");
                                                    free(new_sec_index_first_part);
                                                    free(new_check1_array);
                                                    free(index_array_first);
                                                    return 1;
                                                }
                                                new_sec_index_first_part[new_array_size] = sec_index_first_part[col][0];  // 按列存储
                                                if(current_tree_state == 1)
                                                {
                                                    new_check1_array[new_array_size] = 0;
                                                    current_state = 1;
                                                }
                                                if(current_tree_state == 2)
                                                {
                                                    new_check1_array[new_array_size] = 1;
                                                    current_state = 2;
                                                }
                                                index_array_first[index_array_size] = col;
                                                index_array_size++;
                                                new_array_size++;
                                            }  
                                        }
                                    }

                                    printf("Stored Indexes in updated_index_array:\n");
                                    for (int i = 0; i < updated_index_array_size; i++) {
                                        printf("Index[%d] = %d\n", i, updated_index_array[i]);
                                    }

                                    printf("val_0 == -1, stored indexes in index_array_first:\n");
                                    for (int i = 0; i < index_array_size; i++) {
                                        printf("Index[%d] = %d\n", i, index_array_first[i]);
                                    }
                                    // For certain elements in u[0, k1], construct attack ciphertexts using (l11, l12, -l13)

                                    memcpy(sec_index_first_part, updated_sec_index_first_part, updated_array_size * sizeof(*updated_sec_index_first_part));
                                    check1_array = updated_check1_array;
                                    start_idx = 0;
                                    end_idx = updated_array_size;
                                    array_size = updated_array_size;  
                                    vote_limit=1;

                                    free(get_er_decrypt_array_all);
                                    get_er_decrypt_array_all = (int *)malloc(array_size * sizeof(int));
                                    if (get_er_decrypt_array_all == NULL) {
                                        printf("Memory allocation failed\n");
                                        return 1;
                                    }
                             
                                    get_oracle_response(ct, ss, ss1, pk, sk,
                                                        sec_index_first_part, 
                                                        get_er_decrypt_array_all, check1_array, 
                                                        param_list1_index, param_list2_index, 
                                                        start_idx, end_idx, 
                                                        vote_limit, &overall_oracle_count);
                                    
                                    printf("For new sec_index, actual oracle response is...\n");
                                    for (int i = 0; i < array_size; i++) {
                                        printf("%d,", get_er_decrypt_array_all[i]);
                                    }
                                    printf("\n");
                                    param_value_to_fill = 0;

                                    // Store the obtained oracle response in the array oracle_response_1,
                                    // and store the corresponding p_pos and p_neg of this ciphertext into Pos_neg_index_first
                                    for (int i = 0; i < updated_array_size; i++) 
                                    {
                                        int target_index = updated_index_array[i];
                                        if (oracle_response_1[target_index] == NULL) 
                                        {
                                            oracle_response_1[target_index] = (int *)malloc(1 * sizeof(int));
                                            oracle_response_1_col_sizes[target_index] = 1;
                                        } else {
                                            int current_col_size = oracle_response_1_col_sizes[target_index];
                                            oracle_response_1[target_index] = (int *)realloc(oracle_response_1[target_index], (current_col_size + 1) * sizeof(int));
                                            if (oracle_response_1[target_index] == NULL) {
                                                printf("Memory reallocation failed for oracle_response_1[%d]\n", target_index);
                                                return 1;
                                            }
                                            oracle_response_1_col_sizes[target_index] = current_col_size + 1;
                                        }

                                        int current_col_size = oracle_response_1_col_sizes[target_index];
                                        oracle_response_1[target_index][current_col_size - 1] = get_er_decrypt_array_all[i];

                                        if (Pos_neg_index_first[target_index] == NULL) {
                                            Pos_neg_index_first[target_index] = (int *)malloc(1 * sizeof(int));
                                            Pos_neg_index_first_col_sizes[target_index] = 1;
                                        } else {
                                            int current_col_size = Pos_neg_index_first_col_sizes[target_index];
                                            Pos_neg_index_first[target_index] = (int *)realloc(Pos_neg_index_first[target_index], (current_col_size + 1) * sizeof(int));
                                            if (Pos_neg_index_first[target_index] == NULL) {
                                                printf("Memory reallocation failed for Pos_neg_index_first[%d]\n", target_index);
                                                return 1;
                                            }
                                            Pos_neg_index_first_col_sizes[target_index] = current_col_size + 1;
                                        }

                                        current_col_size = Pos_neg_index_first_col_sizes[target_index];
                                        Pos_neg_index_first[target_index][current_col_size - 1] = param_value_to_fill;
                                    }

                                    printf("Oracle response 1, column-wise output:\n");
                                    for (int col = 0; col < oracle_response_1_capacity; col++) {
                                        printf("Column %d: ", col);
                                        for (int row = 0; row < oracle_response_1_col_sizes[col]; row++) {
                                            printf("%d ", oracle_response_1[col][row]);
                                        }
                                        printf("\n");
                                    }

                                    printf("Pos_neg_index_first, column-wise output:\n");
                                    for (int col = 0; col < oracle_response_1_capacity; col++) {
                                        printf("Column %d: ", col);
                                        for (int row = 0; row < Pos_neg_index_first_col_sizes[col]; row++) {
                                            printf("%d ", Pos_neg_index_first[col][row]);
                                        }
                                        printf("\n");
                                    }
                                    // Determine the next action based on the current oracle response value;
                                    // Check the values of the oracle response obtained via (ds0, ds0_invers) or (ds1, ds1_invers) for indices u ∈ [0, k1]
                                    memcpy(sec_index_first_part, backup_sec_index_first_part, u_row_list_size * sizeof(*sec_index_first_part));
                                    zeros= 0;
                                    
                                    for (int i = 0; i < array_size; i++) 
                                    {
                                        int target_index = updated_index_array[i];
                                        if (target_index >= u_row_list_size || target_index < 0) {
                                            printf("Error: target_index %d is out of bounds\n", target_index);
                                            return 1;
                                        }

                                        int val_row_0 = oracle_response_1[target_index][0];
                                        int val_row_1 = oracle_response_1[target_index][1];
                                        if (current_tree_state == 1 && val_row_0 == 0 && val_row_1 == 0) {
                                            zeros++;  
                                        }                                
                                        if (current_tree_state == 2 && val_row_0 == -1 && val_row_1 == -1) {
                                            zeros++;  
                                        }

                                        if ((current_tree_state == 1 && val_row_0 == 0 && val_row_1 == -1) || (current_tree_state == 2 && val_row_0 == -1 && val_row_1 == 0)) 
                                        {
                                            new_sec_index_first_part = (int *)realloc(new_sec_index_first_part, (new_array_size + 1) * sizeof(int));
                                            new_check1_array = (int *)realloc(new_check1_array, (new_array_size + 1) * sizeof(int));
                                            index_array_first = (int *)realloc(index_array_first, (index_array_size + 1) * sizeof(int));

                                            if (new_sec_index_first_part == NULL || new_check1_array == NULL || index_array_first == NULL) {
                                                printf("Memory allocation failed during resizing.\n");
                                                free(new_sec_index_first_part);
                                                free(new_check1_array);
                                                free(index_array_first);
                                                return 1;
                                            }

                                            new_sec_index_first_part[new_array_size] = sec_index_first_part[target_index][0];
                                            if(current_tree_state == 1)
                                            {
                                                new_check1_array[new_array_size] = 1;
                                            }
                                            if(current_tree_state == 2)
                                            {
                                                new_check1_array[new_array_size] = 0;
                                            }
                                            index_array_first[index_array_size] = target_index;

                                            index_array_size++;
                                            new_array_size++;
                                        }
                                    }
                                    // If the obtained distinguisher seems abnormal, invert it and recreate the base ciphertext
                                    if (zeros > 200 || zeros < 110) 
                                    {                                        
                                        printf("Condition met, jumping to rej...\n");
                                        goto rej;
                                    }
                                    else
                                    {                                  
                                        // Now compute the actual oracle response for (l_11, l_12, l_13) with u1 in [k2, 760]
                                        start_idx = weight_list_k2;
                                        end_idx = p;
                                        array_size = end_idx - start_idx;
                                        vote_limit=1;
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
                                                            sec_index_first_part, 
                                                            get_er_decrypt_array_all, check1_array, 
                                                            param_list1_index, param_list2_index, 
                                                            start_idx, end_idx, 
                                                            vote_limit, &overall_oracle_count);
                                        printf("For new sec_index, actual oracle response is...\n");
                                        for (int i = 0; i < array_size; i++) {
                                            printf("%d,", get_er_decrypt_array_all[i]);
                                        }
                                        printf("\n"); 

                                        // Store the obtained oracle response in the array oracle_response_1,
                                        // and store the corresponding p_pos and p_neg of this ciphertext into Pos_neg_index_first
                                        param_value_to_fill = 0;  

                                        for (int idx1 = start_idx; idx1 < end_idx; idx1++) 
                                        {
                                            if (oracle_response_1[idx1] == NULL) {
                                                oracle_response_1[idx1] = (int *)malloc(1 * sizeof(int));  
                                                oracle_response_1_col_sizes[idx1] = 1;  
                                                if (oracle_response_1[idx1] == NULL) {
                                                    printf("Memory allocation failed for oracle_response_1[%d]\n", idx1);
                                                    return 1;
                                                }
                                            } else {
                                                oracle_response_1_col_sizes[idx1] = 1;
                                            }
                                            oracle_response_1[idx1][0] = get_er_decrypt_array_all[idx1 - start_idx];

                                            if (Pos_neg_index_first[idx1] == NULL) {
                                                Pos_neg_index_first[idx1] = (int *)malloc(1 * sizeof(int));  
                                                Pos_neg_index_first_col_sizes[idx1] = 1;  
                                                if (Pos_neg_index_first[idx1] == NULL) {
                                                    printf("Memory allocation failed for Pos_neg_index_first[%d]\n", idx1);
                                                    return 1;
                                                }
                                            } else {
                                                Pos_neg_index_first_col_sizes[idx1] = 1;
                                            }

                                            Pos_neg_index_first[idx1][0] = param_value_to_fill;
                                        }

                                        printf("Contents of oracle_response_1 (Column-wise): oracle_response_1_size is %d\n", oracle_response_1_size);
                                        for (int col = 0; col < oracle_response_1_capacity; col++) {
                                            printf("Column %d: ", col);
                                            for (int row = 0; row < oracle_response_1_col_sizes[col]; row++) {
                                                printf("%d ", oracle_response_1[col][row]);
                                            }
                                            printf("\n");
                                        }

                                        printf("Contents of Pos_neg_index_first (Column-wise):\n");
                                        for (int col = 0; col < oracle_response_1_capacity; col++) {
                                            printf("Column %d: ", col);
                                            for (int row = 0; row < Pos_neg_index_first_col_sizes[col]; row++) {
                                                printf("%d ", Pos_neg_index_first[col][row]);
                                            }
                                            printf("\n");
                                        }

                                        // Check the values of the oracle response obtained via (ds0) or (ds1) for indices u ∈ [k2, p]
                                        for (int col = start_idx; col < oracle_response_1_capacity; col++)  // 遍历每一列
                                        {
                                            if (oracle_response_1_col_sizes[col] > 0)
                                            {
                                                int val_0 = oracle_response_1[col][0];  
                                                if((current_tree_state == 1 && val_0 == 0)|| (current_tree_state == 2 && val_0 == -1))
                                                {
                                                    adi_sec_index_first_part = (int *)realloc(adi_sec_index_first_part, (adi_array_size + 1) * sizeof(int));
                                                    adi_check1_array = (int *)realloc(adi_check1_array, (adi_array_size + 1) * sizeof(int));
                                                    adi_index_array_first = (int *)realloc(adi_index_array_first, (adi_index_array_size + 1) * sizeof(int));

                                                    if (adi_sec_index_first_part == NULL || adi_check1_array == NULL || adi_index_array_first == NULL) {
                                                        printf("Memory allocation failed during resizing.\n");
                                                        free(adi_sec_index_first_part);
                                                        free(adi_check1_array);
                                                        free(adi_index_array_first);
                                                        return 1;
                                                    }

                                                    adi_sec_index_first_part[adi_array_size] = sec_index_first_part[col][0];  
                                                    adi_check1_array[adi_array_size] = 1;

                                                    adi_index_array_first[adi_index_array_size] = col;
                                                    adi_index_array_size++;
                                                    adi_array_size++;              
                                                }                                        
                                                else if ((current_tree_state == 1 && val_0 == -1) || (current_tree_state == 2 && val_0 == 0)) 
                                                {
                                                    new_sec_index_first_part = (int *)realloc(new_sec_index_first_part, (new_array_size + 1) * sizeof(int));
                                                    new_check1_array = (int *)realloc(new_check1_array, (new_array_size + 1) * sizeof(int));
                                                    index_array_first = (int *)realloc(index_array_first, (index_array_size + 1) * sizeof(int));  

                                                    if (new_sec_index_first_part == NULL || new_check1_array == NULL || index_array_first == NULL) {
                                                        printf("Memory allocation failed during resizing.\n");
                                                        free(new_sec_index_first_part);
                                                        free(new_check1_array);
                                                        free(index_array_first);
                                                        return 1;
                                                    }

                                                    new_sec_index_first_part[new_array_size] = sec_index_first_part[col][0]; 
                                                    if(current_tree_state == 1)
                                                    {
                                                        new_check1_array[new_array_size] = 0;
                                                    }
                                                    if(current_tree_state == 2)
                                                    {
                                                        new_check1_array[new_array_size] = 1;
                                                    }

                                                    index_array_first[index_array_size] = col;
                                                    index_array_size++;
                                                    new_array_size++;
                                                }  
                                            }
                                        }

                                        // For certain elements in u[k2, p], construct attack ciphertexts using (l11, l12, -l13)
                                        memcpy(sec_index_first_part, adi_sec_index_first_part, adi_array_size * sizeof(*adi_sec_index_first_part));
                                        start_idx = 0;
                                        check1_array = adi_check1_array;  
                                        end_idx = adi_array_size;
                                        array_size = adi_array_size;     
                                        vote_limit=1;

                                        
                                        free(get_er_decrypt_array_all);
                                        get_er_decrypt_array_all = (int *)malloc(array_size * sizeof(int));
                                        if (get_er_decrypt_array_all == NULL) {
                                            printf("Memory allocation failed\n");
                                            return 1;
                                        }
                                        
                                        get_oracle_response(ct, ss, ss1, pk, sk,
                                                            sec_index_first_part, 
                                                            get_er_decrypt_array_all, check1_array, 
                                                            param_list1_index, param_list2_index, 
                                                            start_idx, end_idx, 
                                                            vote_limit, &overall_oracle_count);
                                        
                                        printf("for adi sec_index Actual oracle response is...\n");
                                        for (int i = 0; i < array_size; i++) {
                                            printf("%d,", get_er_decrypt_array_all[i]);
                                        }
                                        printf("\n");
                                       
                                        param_value_to_fill = 0;  

                                        // Store the obtained oracle response in the array oracle_response_1,
                                        // and store the corresponding p_pos and p_neg of this ciphertext into Pos_neg_index_first
                                        for (int i = 0; i < adi_array_size; i++) 
                                        {
                                            int target_index = adi_index_array_first[i];  
                                            if (oracle_response_1[target_index] == NULL) {
                                                oracle_response_1[target_index] = (int *)malloc(1 * sizeof(int));
                                                oracle_response_1_col_sizes[target_index] = 1;
                                            } else {
                                                int current_col_size = oracle_response_1_col_sizes[target_index];
                                                oracle_response_1[target_index] = (int *)realloc(oracle_response_1[target_index], (current_col_size + 1) * sizeof(int));
                                                if (oracle_response_1[target_index] == NULL) {
                                                    printf("Memory reallocation failed for oracle_response_1[%d]\n", target_index);
                                                    return 1;
                                                }
                                                oracle_response_1_col_sizes[target_index] = current_col_size + 1;
                                            }

                                            int current_col_size = oracle_response_1_col_sizes[target_index];
                                            oracle_response_1[target_index][current_col_size - 1] = get_er_decrypt_array_all[i];

                                            if (Pos_neg_index_first[target_index] == NULL) {
                                                Pos_neg_index_first[target_index] = (int *)malloc(1 * sizeof(int));
                                                Pos_neg_index_first_col_sizes[target_index] = 1;
                                            } else {
                                                int current_col_size = Pos_neg_index_first_col_sizes[target_index];
                                                Pos_neg_index_first[target_index] = (int *)realloc(Pos_neg_index_first[target_index], (current_col_size + 1) * sizeof(int));
                                                if (Pos_neg_index_first[target_index] == NULL) {
                                                    printf("Memory reallocation failed for Pos_neg_index_first[%d]\n", target_index);
                                                    return 1;
                                                }

                                                Pos_neg_index_first_col_sizes[target_index] = current_col_size + 1;
                                            }

                                            current_col_size = Pos_neg_index_first_col_sizes[target_index];
                                            Pos_neg_index_first[target_index][current_col_size - 1] = param_value_to_fill;
                                        }

                                        printf("Oracle response 1, column-wise output:\n");
                                        for (int col = 0; col < u_row_list_size; col++) {
                                            printf("Column %d: ", col);
                                            for (int row = 0; row < oracle_response_1_col_sizes[col]; row++) {
                                                printf("%d ", oracle_response_1[col][row]);
                                            }
                                            printf("\n");
                                        }

                                        printf("Pos_neg_index_first, column-wise output:\n");
                                        for (int col = 0; col < u_row_list_size; col++) {
                                            printf("Column %d: ", col);
                                            for (int row = 0; row < Pos_neg_index_first_col_sizes[col]; row++) {
                                                printf("%d ", Pos_neg_index_first[col][row]);
                                            }
                                            printf("\n");
                                        }

                                        // Determine the next action based on the current oracle response value;
                                        // Check the values of the oracle response obtained via (ds0, ds0_invers) or (ds1, ds1_invers) for indices u ∈ [k2, p]
                                        memcpy(sec_index_first_part, backup_sec_index_first_part, u_row_list_size * sizeof(*sec_index_first_part));

                                        for (int i = 0; i < array_size; i++) 
                                        {
                                            int target_index = adi_index_array_first[i];
                                            if (target_index >= u_row_list_size || target_index < 0) {
                                                printf("Error: target_index %d is out of bounds\n", target_index);
                                                return 1;
                                            }
                                            int val_row_0 = oracle_response_1[target_index][0];
                                            int val_row_1 = oracle_response_1[target_index][1];
                                            if (current_tree_state == 1 && val_row_0 == 0 && val_row_1 == 0) {
                                                zeros++;  
                                            }
                                            
                                            if (current_tree_state == 2 && val_row_0 == -1 && val_row_1 == -1) {
                                                zeros++;  
                                            }

                                            if ((current_tree_state == 1 && val_row_0 == 0 && val_row_1 == -1) || (current_tree_state == 2 && val_row_0 == -1 && val_row_1 == 0)) 
                                            {

                                                new_sec_index_first_part = (int *)realloc(new_sec_index_first_part, (new_array_size + 1) * sizeof(int));
                                                new_check1_array = (int *)realloc(new_check1_array, (new_array_size + 1) * sizeof(int));
                                                index_array_first = (int *)realloc(index_array_first, (index_array_size + 1) * sizeof(int));

                                                if (new_sec_index_first_part == NULL || new_check1_array == NULL || index_array_first == NULL) {
                                                    printf("Memory allocation failed during resizing.\n");
                                                    free(new_sec_index_first_part);
                                                    free(new_check1_array);
                                                    free(index_array_first);
                                                    return 1;
                                                }

                                                new_sec_index_first_part[new_array_size] = sec_index_first_part[target_index][0];
                                                if(current_tree_state == 1)
                                                {
                                                    new_check1_array[new_array_size] = 1;
                                                }
                                                if(current_tree_state == 2)
                                                {
                                                    new_check1_array[new_array_size] = 0;
                                                }
                                                index_array_first[index_array_size] = target_index;

                                                index_array_size++;
                                                new_array_size++;
                                            }
                                        }


                                        // Next, attempt to find ds2
                                        use_new_sec_index = true;
                                        // Use the parameter list corresponding to ds
                                        param_which_list = 1;
                                        // Start trying from the first element of the list
                                        param_list2_index = 0; 
                                                                        
                                        if(current_tree_state == 1)
                                        {        
                                            // Binary tree corresponding to ds0  
                                            coding_arr_1[tree_size_1++] = create_node(true, 1);
                                            coding_arr_1[tree_size_1++] = create_node(false, -1);                           
                                            coding_arr_1[tree_size_1++] = create_node(true, 2);
                                            coding_arr_1[tree_size_1++] = NULL;
                                            coding_arr_1[tree_size_1++] = create_node(false, -2); 

                                            current_state = 1;              
                                        }
                                        if(current_tree_state == 2)
                                        {      
                                            // Binary tree corresponding to ds1                                    
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

                            }     
                          
                            if (current_dis == 2)
                            {    
                                // If ds2 is obtained, 
                                // Store the obtained oracle response in the array oracle_response_1,
                                // and store the corresponding p_pos and p_neg of this ciphertext into Pos_neg_index_first
                                for (int i = 0; i < new_array_size; i++) 
                                {
                                    int target_index = index_array_first[i];  
                                    if (oracle_response_1[target_index] == NULL) {
                                        oracle_response_1[target_index] = (int *)malloc(1 * sizeof(int));
                                        oracle_response_1_col_sizes[target_index] = 1;
                                    } else {
                                        int current_col_size = oracle_response_1_col_sizes[target_index];
                                        oracle_response_1[target_index] = (int *)realloc(oracle_response_1[target_index], (current_col_size + 1) * sizeof(int));
                                        if (oracle_response_1[target_index] == NULL) {
                                            printf("Memory reallocation failed for oracle_response_1[%d]\n", target_index);
                                            return 1;
                                        }
                                        
                                        oracle_response_1_col_sizes[target_index] = current_col_size + 1;
                                    }
                                    
                                    int current_col_size = oracle_response_1_col_sizes[target_index];
                                    oracle_response_1[target_index][current_col_size - 1] = get_er_decrypt_array_all[i];
                                }

                                printf("Oracle response 1, column-wise output:\n");
                                for (int col = 0; col < u_row_list_size; col++) {
                                    printf("Column %d: ", col);
                                    for (int row = 0; row < oracle_response_1_col_sizes[col]; row++) {
                                        printf("%d ", oracle_response_1[col][row]);
                                    }
                                    printf("\n");
                                }
                                param_value_to_fill = 0;
 
                                for (int i = 0; i < new_array_size; i++) 
                                {
                                    int target_index = index_array_first[i];  
                                    int current_col_size = Pos_neg_index_first_col_sizes[target_index];

                                    if (check1_array[i] == 0) {
                                        
                                        Pos_neg_index_first[target_index] = (int *)realloc(Pos_neg_index_first[target_index], (current_col_size + 1) * sizeof(int));
                                        if (Pos_neg_index_first[target_index] == NULL) {
                                            printf("Memory reallocation failed for Pos_neg_index_first[%d]\n", target_index);
                                            return 1;
                                        }
                                        Pos_neg_index_first[target_index][current_col_size] = param_value_to_fill;
                                    } 
                                    else if (check1_array[i] == 1) {
                                        
                                        Pos_neg_index_first[target_index] = (int *)realloc(Pos_neg_index_first[target_index], (current_col_size + 1) * sizeof(int));
                                        if (Pos_neg_index_first[target_index] == NULL) {
                                            printf("Memory reallocation failed for Pos_neg_index_first[%d]\n", target_index);
                                            return 1;
                                        }
                                        Pos_neg_index_first[target_index][current_col_size] = param_value_to_fill;
                                    }

                                    Pos_neg_index_first_col_sizes[target_index] = current_col_size + 1;
                                }

 
                                printf("Pos_neg_index_first, column-wise output:\n");
                                for (int col = 0; col < u_row_list_size; col++) {
                                    printf("Column %d: ", col);
                                    for (int row = 0; row < Pos_neg_index_first_col_sizes[col]; row++) {
                                        printf("%d ", Pos_neg_index_first[col][row]);
                                    }
                                    printf("\n");
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
                            // If the expected distinguisher cannot be successfully found for the current base ciphertext, jump to regenerate the base ciphertext                        
                            printf("Unsuccessful search, jumping to rej...\n");
                            goto rej;                        
                        }                        
                       
                    }

                    // For checking purposes only, save the corresponding parameters used to construct the attack ciphertext
                    if (cond_prob_loop == 0) 
                    {
                        FILE *parameters_list_for_attack_ciphertext_file = fopen("parameters_list_for_attack_ciphertext.bin", "a");
                        if (parameters_list_for_attack_ciphertext_file == NULL) {
                            printf("Failed to open file: parameters_list_for_attack_ciphertext.bin\n");
                            return 1;
                        }
                                                    
                        int indices[4];                                                 
                        indices[0] = 0; indices[1] = 1;
                        
                        fprintf(parameters_list_for_attack_ciphertext_file, "for NO_TESTS is %d single_used_param:\n", pq);

                      
                        for (int i = 0; i < 2; i++) {  
                            int index = indices[i];
                            fprintf(parameters_list_for_attack_ciphertext_file, "Record %d: param_which_list = %d, param_list1_index = %d, param_list2_index = %d\n",
                                    index,
                                    param_records[index].param_which_list,
                                    param_records[index].param_list1_index,
                                    param_records[index].param_list2_index);
                        }
 
                        fclose(parameters_list_for_attack_ciphertext_file);
                    }

                    use_new_sec_index = false;
                                                  
                    // Generate the required internal coding list based on the oracle response
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

                    int out[MAX_BIT_LENGTH];  
                    int size;                 
                    int sum_weight = 4;     

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

                    // Store the obtained oracle response solely for verification purposes
                    FILE *actual_oracle_response_file = fopen("actual_oracle_response.txt", "a");
                    if (actual_oracle_response_file == NULL) {
                        printf("Failed to open file: actual_oracle_response.txt\n");
                        return 1;
                    }

                    fprintf(actual_oracle_response_file, "for NO_TESTS is %d cond_prob_loop is %d actual oracle response for each test:\n", pq, cond_prob_loop);

                    int actual_loop_count = 1;
                    if (cond_prob_loop == 0) 
                    {
                        for (int col = 0; col < p; col++) {
                            fprintf(actual_oracle_response_file, "actual_loop_count:%d, actual_oracle_response:", actual_loop_count++);
                            
                            for (int row = 0; row < oracle_response_1_col_sizes[col]; row++) {
                                fprintf(actual_oracle_response_file, "%d ", oracle_response_1[col][row]);
                            }
                            fprintf(actual_oracle_response_file, "\n");
                        }                        

                    } 

                    fclose(actual_oracle_response_file);
                
                    // Store the theoretical oracle response solely for verification purposes; debugging information is used here
                    int beta_u;
                    char beta_u_expr[256] = ""; 
                    int coeff_now =0;
                    int coeff_now_1, coeff_now_2;                    
                    theoretical_loop_count = 0; 
                                                         
                    memcpy(sec_index_first_part, backup_sec_index_first_part, u_row_list_size * sizeof(*sec_index_first_part));
                    
                    beta_u_and_theoretical_oracle_response_file = fopen("beta_u_and_theoretical_oracle_response.bin", "a");
                    if (beta_u_and_theoretical_oracle_response_file == NULL) {
                        printf("Failed to open file: beta_u_and_theoretical_oracle_response.bin\n");
                        return 1;
                    }
                   
                    fprintf(beta_u_and_theoretical_oracle_response_file, "for NO_TESTS is %d cond_prob_loop is %d beta_u and theoretical oracle response for each test:\n",pq, cond_prob_loop);

                    fclose(beta_u_and_theoretical_oracle_response_file);

                    if (cond_prob_loop == 0)
                    {                      
                        for (int idx1 = 0; idx1 < p; idx1++) 
                        {
                            theoretical_loop_count++; 
                            beta_u = 0;
                            beta_u_expr[0] = '\0'; 
                            int beta_coeff_count = 0; 
                            
                            for(int j = 0; j < 1; j++) 
                            {
                                int uj = sec_index_first_part[idx1][j];
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

                            sample_coef_with_simple_oracle(oracle, beta_u, coding_tree_1, out, &size);  

                            printf("theoretical_loop_count: %d, beta_u: %d, oracle output: (", theoretical_loop_count, beta_u);
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

                    }
                    
                    // Now, calculate the conditional probabilities based on the obtained oracle response, 
                    // and pass them as input to the LDPC decoder to recover the secret key
                    int w1 = 286;
                    long double Pr[new_table_row_size];
                    FILE *file1 = NULL;
                    FILE *file2 = NULL;
                   
                    char command[512];
                 
                    long double **oracle_accuracy = init_oracle_accuracy();
                    FalsePositiveNegativePositionalOracle *pr_oracle = create_fp_fn_oracle(oracle_accuracy, 1);
                    
                    for (int coll_index = 0; coll_index < p; coll_index++)
                    {     
                        
                        char filename[256];
                        #if (DO_PRINT == 1)
                        sprintf(filename, "When 1 %d for alpha_u_and_conditional_probabilities.bin", coll_index);
                       
                        if (cond_prob_loop == 0) {
                            file1 = fopen(filename, "w");
                        } else if (cond_prob_loop > 0) {
                            file1 = fopen(filename, "a");
                        }

                        if (file1 == NULL) {
                            printf("Failed to open file: %s\n", filename);
                            continue;
                        }

                       
                        #endif
                        
                        char beta_u_expr[256] = "";

                        int coeff_now_att;
                        int coeff_now_1_att, coeff_now_2_att;
                        
                        if(cond_prob_loop == 0)
                        {
                            for (int idx1 = 0; idx1 < p; idx1++) 
                            {

                                beta_u_expr[0] = '\0';                        
                                int beta_coeff_count = 0; 
                                for(int j = 0; j < 1; j++) 
                                {
                                    int uj = sec_index_first_part[idx1][j];
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
                                // Extract the idx1-th column of oracle_response_1
                                int *y = NULL;
                                int y_len = 0;
                                // **Find the size of the idx1-th column**
                                // Use oracle_response_1_col_sizes[idx1] to get the length of the idx1-th column
                                if (oracle_response_1[idx1] != NULL) 
                                {
                                    y_len = oracle_response_1_col_sizes[idx1];  
                                    y = (int *)malloc(y_len * sizeof(int));
                                    if (y == NULL) 
                                    {
                                        printf("Memory allocation failed for y\n");
                                        return 1;
                                    }

                                    // Copy the contents of the idx1-th column of oracle_response_1 into y
                                    for (int i = 0; i < y_len; i++) 
                                    {
                                        y[i] = oracle_response_1[idx1][i];
                                    }
                                } 
                                else 
                                {
                                    printf("No data in oracle_response_1[%d]\n", idx1);  // 打印信息：没有数据
                                }

                                // Extract the idx1-th column of Pos_neg_index_first
                                int *y_prob = NULL;
                                int y_prob_len = 0;

                                // **Find the size of the idx1-th column**
                                // Use Pos_neg_index_first_col_sizes[idx1] to get the length of the idx1-th column
                                if (Pos_neg_index_first[idx1] != NULL) 
                                {
                                    y_prob_len = Pos_neg_index_first_col_sizes[idx1];  
                                    y_prob = (int *)malloc(y_prob_len * sizeof(int));
                                    if (y_prob == NULL) 
                                    {
                                        printf("Memory allocation failed for y_prob\n");
                                        return 1;
                                    }

                                    for (int i = 0; i < y_prob_len; i++) 
                                    {
                                        y_prob[i] = Pos_neg_index_first[idx1][i];
                                    }
                                  
                                } 
                                else 
                                {
                                    printf("No data in Pos_neg_index_first[%d]\n", idx1);  
                                }
                                long double conditional_probabilities_1[new_table_row_size] = {0.0};
                                if (y != NULL && y_prob != NULL) 
                                {
                                    for (int s = -sum_weight; s <= sum_weight; s++) 
                                    {
                                        long double pr_x_y = pr_cond_xy_adaptive(s, y, y_len, pr_oracle, Pr, sum_weight, coding_tree_1, y_prob);  
                                        conditional_probabilities_1[s + sum_weight] = pr_x_y;  
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

                                // Write conditional probabilities to file
                                for (int i = 0; i < new_table_row_size; i++) 
                                {
                                    if (i > 0) 
                                    {
                                        fprintf(file4, ",");
                                    }
                                    fprintf(file4, "%.20Lf", conditional_probabilities_1[i]);
                                }
                                fprintf(file4, "\n");
                               
                                // fprintf(file_test, "%s\n", beta_u_expr);
                                // for (int i = 0; i < new_table_row_size; i++) 
                                // {
                                //     if (i > 0) 
                                //     {
                                //         fprintf(file_test, ",");
                                //     }
                                //     fprintf(file_test, "%.20Lf", conditional_probabilities_1[i]);
                                // }
                                // fprintf(file_test, "\n");

                                #endif
                            
                            }

                        }
                           
                        #if (DO_PRINT == 1)
                        if (file1 != NULL) fclose(file1);
                        // if (file_test != NULL) fclose(file_test);

                        // if (file2 != NULL) fclose(file2); 
                        #endif
 
                        if(cond_prob_loop == 0)
                        {
                        
                            // FILE *after_change_ldpc_output_file = fopen("after_change_ldpc.bin", "a");
                            // if (after_change_ldpc_output_file == NULL) {
                            //     printf("Failed to open file: ldpc.bin\n");
                            //     return 1;
                            // }
                            
                            // fprintf(after_change_ldpc_output_file, "pq_counter: %d,cond_prob_loop: %d,collision_index: %d,collision_value: %d,", pq, cond_prob_loop, collision_index, collision_value);
                            // fprintf(after_change_ldpc_output_file, "\n");

                            // fclose(after_change_ldpc_output_file);

                            char command[1024];  
                            // Call the Python script to invoke the LDPC decoder            
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

                            // ================= Read data from the file fprime_weight.txt ===================
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

                            fprintf(secret_weight_file, "for NO_TESTS is %d cond_prob_loop is %d Assuming collision index is %d\n", pq, cond_prob_loop, coll_index);

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
                            printf("end\n");
                            

                            // ================== Read the LDPC output and store the obtained secret key into ldpc_fprime_output.txt ==================
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
                    }
                                     
                   cond_prob_loop++; 
                } 

                // ================== Record the number of oracle calls required for each key recovery ==================
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
                

                if (oracle_response_1 != NULL) {  
                    for (int i = 0; i < oracle_response_1_size; i++) {
                        if (oracle_response_1[i] != NULL) {  
                            free(oracle_response_1[i]);  
                            oracle_response_1[i] = NULL;  
                        }
                    }
                    free(oracle_response_1);  
                    oracle_response_1 = NULL; 
                }

                if (oracle_response_1_col_sizes != NULL) { 
                    free(oracle_response_1_col_sizes); 
                    oracle_response_1_col_sizes = NULL;  
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
 
    free(c_value_for_attack_1_1);
    free(c_value_for_attack_1_2);
    free(c_value_for_attack_1_3);
                
    #endif

    return KAT_SUCCESS;
}
