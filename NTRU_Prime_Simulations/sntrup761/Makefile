# kat_kem.rsp: kat
# 	./run
#

# kat: Makefile nist/kat_kem.c nist/rng.c nist/rng.h randombytes.h Decode.c Encode.c int32.c kem.c uint32.c     subroutines/crypto_hash_sha512.c subroutines/crypto_sort_uint32.c subroutines/crypto_stream_aes256ctr.c
# 	./build

test_ex: Makefile nist/test.c nist/rng.c nist/rng.h randombytes.h crypto_kem.h attack_parameters.h Decode.c Encode.c int32.c kem_mod.c uint32.c     subroutines/crypto_hash_sha512.c subroutines/crypto_sort_uint32.c subroutines/crypto_stream_aes256ctr.c
	gcc -O3 -march=native -std=c99 -mtune=native -Wall -I. -Isubroutines "-DCRYPTO_NAMESPACE(x)=x" -o test_ex nist/test.c nist/rng.c Decode.c Encode.c int32.c kem_mod.c uint32.c     subroutines/crypto_hash_sha512.c subroutines/crypto_sort_uint32.c subroutines/crypto_stream_aes256ctr.c  -lcrypto -ldl -lm

clean:
	rm -rf test_ex

# # Makefile

# # 默认目标：在编译之前先执行 setup-environment.sh
# all: setup test_ex

# # setup 目标：调用 setup-environment.sh 脚本
# setup:
# 	@echo "Running setup-environment.sh..."
# 	@bash /home/fei/C/SCA_Assisted_CCA_on_NTRU/NTRU_Prime/PC_Oracle_based_SCA/Attack_Simulations/sntrup761/SCA-LDPC/simulate-with-python/setup-environment.sh

# test_ex: Makefile nist/test.c nist/rng.c nist/rng.h randombytes.h crypto_kem.h attack_parameters.h Decode.c Encode.c int32.c kem_mod.c uint32.c subroutines/crypto_hash_sha512.c subroutines/crypto_sort_uint32.c subroutines/crypto_stream_aes256ctr.c
# 	gcc -O3 -march=native -std=c99 -mtune=native -Wall -I. -Isubroutines "-DCRYPTO_NAMESPACE(x)=x" -o test_ex nist/test.c nist/rng.c Decode.c Encode.c int32.c kem_mod.c uint32.c subroutines/crypto_hash_sha512.c subroutines/crypto_sort_uint32.c subroutines/crypto_stream_aes256ctr.c -lcrypto -ldl -lm

# clean:
# 	rm -rf test_ex