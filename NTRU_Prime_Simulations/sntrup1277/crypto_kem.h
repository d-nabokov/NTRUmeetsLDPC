#ifndef crypto_kem_H
#define crypto_kem_H

#include "crypto_kem_sntrup1277.h"
#include "int8.h"
#include "int16.h"
#include "int32.h"
#include "uint16.h"
#include "uint32.h"
#include "crypto_sort_uint32.h"
#include "Encode.h"
#include "Decode.h"
#include <math.h>
#include "params.h"

#define crypto_kem_keypair crypto_kem_sntrup1277_keypair
#define crypto_kem_enc crypto_kem_sntrup1277_enc
#define crypto_kem_dec crypto_kem_sntrup1277_dec
#define crypto_kem_PUBLICKEYBYTES crypto_kem_sntrup1277_PUBLICKEYBYTES
#define crypto_kem_SECRETKEYBYTES crypto_kem_sntrup1277_SECRETKEYBYTES
#define crypto_kem_BYTES crypto_kem_sntrup1277_BYTES
#define crypto_kem_CIPHERTEXTBYTES crypto_kem_sntrup1277_CIPHERTEXTBYTES
#define crypto_kem_PRIMITIVE "sntrup1277"

#define count_threshold 1
#define index_threshold 0
#define NO_COEFFS 10

extern int intended_function;
// extern int sec_index;
extern int no_leakage_trials;

#define DO_ATTACK_COLLISION_NEW 1

#define GAP_THRESHOLD_1_1 110
#define GAP_THRESHOLD_1_2 110
#define GAP_THRESHOLD_2_1 200
#define GAP_THRESHOLD_2_2 200
#define C_VALUE_THRESHOLD_1 130
#define C_VALUE_THRESHOLD_2 130

#define TRIALS_FOR_SHUFFLING 1
#define TOTAL_COEFFS_TO_FIND (p-2)

extern int error_now;

extern int succ_flag;
extern int count_ones;
extern int count_plus_ones;
extern int count_minus_ones;
extern int non_zero_f_coeff;
extern int non_zero_g_coeff;
extern int check_for_value;

extern int mul_value;

extern int c_value;
extern int c_value_1;
extern int c_value_2;
extern int c_value_for_attack_1;
extern int c_value_for_attack_2;

extern int c_value_for_leakage;
extern int collision_index;
extern int collision_value;
extern int hw_value;
extern int weight_hh;

extern int m;
extern int n;

typedef int8 small;

extern small er[p];
extern small er_decrypt[p];  // 定义并初始化为 0
extern small global_f[p];
extern small global_g[p];

/* ----- arithmetic mod q */

#define q12 ((q-1)/2)

typedef int16 Fq;

extern Fq global_c_in_encrypt[p];
extern Fq global_c_in_decrypt[p];
extern Fq x_f_array[p];
extern Fq x_g_array[p];
extern Fq cf3[p];
extern Fq f_diff_3[p];

extern Fq cf[p];
extern small e[p];
extern small ev[p];
extern Fq c_copy[p];

#endif
