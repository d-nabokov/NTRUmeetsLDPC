# NTRUmeetsLDPC
The repository will contain the code with experiments for the paper "Enhancing Key-Recovery Chosen-Ciphertext Side-Channel Attacks on NTRU Using LDPC".

## Installation

The code depends on the modified version of the SCA-LDPC framework available at https://github.com/d-nabokov/SCA-LDPC. Our code implicitly assumes that there is already set-up and compiled framework available at `../SCA-LDPC` path. 

Before running the code, make sure to run the following command to activate the virtual environment provided by SCA-LDPC framework
```
source ../SCA-LDPC/python-virtualenv/bin/activate
```

## Running

After setting up the virtual environment, simply run
```
python ntru_simulations.py
```
which will simulate the recovery of 100 keys (one can change parameter TEST_KEYS to analyze up to 1000 keys)

## Note

There was a bug in the usage of LDPC decoder which was spotted later. This negatively affected the conversion to the correct secret key, requiring more traces. Thus, in simulations, instead of `1990` traces for `rho = 0.95` we should get about `1855`. 