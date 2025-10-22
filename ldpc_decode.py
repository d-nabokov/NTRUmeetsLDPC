import inspect
import itertools as it
import logging
import os.path
import re
import sys
from collections import defaultdict
from math import prod

import coloredlogs
import numpy as np
from simulate_rs import (
    DecoderExtendedNTRUW2,
    DecoderExtendedNTRUW4,
    DecoderExtendedNTRUW6,
    DecoderNTRUW2,
    DecoderNTRUW4,
    DecoderNTRUW6,
)

logger = logging.getLogger(__name__.replace("__", ""))
coloredlogs.install(level="INFO", logger=logger)

MOVE_SINGLE_CHECKS_TO_APRIOR = True
USE_EXTENDED_VARIABLES = True

if USE_EXTENDED_VARIABLES:
    B = 2
else:
    B = 1


def extended_variables_indices(indices):
    """Collapse disjoint (x, x+1) pairs."""
    out = []
    i = 0
    n = len(indices)

    while i < n:
        curr = indices[i]

        if i + 1 < n:
            nxt = indices[i + 1]

            # ──────────────────────────────────────────────────────────
            # Special case produced by *u == col_idx* → [p‑1, 0].
            # (0 followed immediately by p‑1).
            # Keep *0* (the first) and consume both.
            if curr == p - 1 and nxt == 0:
                out.append(nxt)
                i += 2
                continue

            # Generic ascending pair [x, x+1] coming from *u > col_idx*.
            # Ensure it is **exactly** a pair and not the beginning of a
            # longer run arising from two different *compute_alpha*
            # calls (e.g. 3, 4, 5).
            if nxt == (curr + 1) % p and not (
                i + 2 < n and (indices[i + 2] % p) == (nxt + 1) % p
            ):
                out.append(nxt)
                i += 2
                continue

        # Fallback: single element produced by *u < col_idx* or second
        # element of a longer run (the first one has already been
        # handled above).
        out.append(curr)
        i += 1

    return out


def resize_pmf(pmf, target_b):
    target_size = 2 * target_b + 1
    if len(pmf) > target_size:
        offset = (len(pmf) - target_size) // 2
        return pmf[offset:-offset]
    elif len(pmf) < target_size:
        offset = (target_size - len(pmf)) // 2
        return [0.0] * offset + pmf + [0.0] * offset
    else:
        return pmf


def process_cond_prob_file(filename, n, check_weight):
    if not os.path.isfile(filename):
        print("File does not exist")
        return None, None

    beta_distrs = list(
        list(sum_secret_distr(f_distr, i + 1).values()) for i in range(check_weight)
    )

    with open(filename, "r") as file:
        lines = file.readlines()

    index_lines = []
    probability_lists = []

    single_check_idxs = []
    single_check_distr = []

    # col_idx = None
    # pred_col_idx = None
    # last_single_index = None

    # read lines in blocks of 2
    for i in range(0, len(lines), 2):
        indices = list(map(int, lines[i].strip().split(",")))
        probabilities = list(map(float, lines[i + 1].strip().split(",")))

        assert len(list(x for x in probabilities if x != 0)) == len(indices) * 2 + 1

        original_indices_len = len(indices)

        if USE_EXTENDED_VARIABLES:
            indices = extended_variables_indices(indices)

        # support the case where extra probabilities are not printed
        if len(probabilities) == len(indices) * 2 + 1 and len(indices) < check_weight:
            offset = check_weight - len(indices)
            probabilities = [0.0] * offset + probabilities + [0.0] * offset

        # if i == 0:
        #     col_idx = indices[0]

        # if pred_col_idx is None and len(indices) == 2:
        #     pred_col_idx = col_idx - last_single_index + 1
        # last_single_index = indices[0]

        if MOVE_SINGLE_CHECKS_TO_APRIOR and len(indices) == 1:
            single_check_idxs.append(indices[0])
            single_check_distr.append(probabilities)
        else:
            index_lines.append(indices)
            # For check variables LDPC should take Pr[y | \sum s_i], but currently
            # in the file we have Pr[\sum s_i | y] which is essentially computed
            # during belief propagation. So, here we divide each probability by
            # Pr[\sum s_i] to get expected value
            probabilities = np.array(probabilities)
            offset = check_weight - original_indices_len
            beta_distr = beta_distrs[original_indices_len - 1]
            for j in range(original_indices_len * 2 + 1):
                probabilities[j + offset] /= beta_distr[j]
            probabilities /= sum(probabilities)
            probability_lists.append(probabilities)

    # Determine the number of parity checks
    num_rows = len(index_lines)
    # Create the matrix with the appropriate size
    matrix = np.zeros((num_rows, n + num_rows), dtype=int)

    # Fill in the ones based on indices and append the negative identity matrix
    for i, indices in enumerate(index_lines):
        for index in indices:
            matrix[i, index] = 1
        matrix[i, n + i] = -1

    return (
        matrix,
        index_lines,
        probability_lists,
        single_check_idxs,
        single_check_distr,
    )


def parse_key_info_file(file_path):
    keys = []
    collisions = []

    p = re.compile("pq_counter: (\d+),inner_test: (\d+)")

    with open(file_path, "r") as f:
        current_key = []
        in_key_section = False
        collision_info = []

        current_counter = None
        for line in f:
            line = line.strip()

            # Look for "pq_counter:"
            if line.startswith("pq_counter:"):
                m = p.match(line)
                pq_counter = int(m[1])
                # If the counter is different, we finished handling and we save it
                if current_counter is None:
                    current_counter = pq_counter
                elif pq_counter != current_counter:
                    current_counter = pq_counter
                    keys.append(current_key)
                    collisions.append(collision_info)
                # Reset variables for new entry
                current_key = []
                in_key_section = False
                collision_info = []

            # Look for "The private key is:"
            elif line == "The private key is:":
                in_key_section = True
                continue  # Skip to next line, where the key starts

            # If we are in the key section, capture the key data
            elif in_key_section:
                if line:  # If the line contains key data
                    # Remove trailing comma and split the key values
                    current_key = [int(x) for x in line.rstrip(",").split(",")]
                    in_key_section = False  # We are done with key section

            # Capture collision index and value
            elif line.startswith("collision_index"):
                index_value = line.split(",")
                collision_index = int(index_value[0].split(":")[1])
                collision_value = int(index_value[1].split(":")[1])
                collision_info.append((collision_index, collision_value))

    # Don't forget to add the last data
    keys.append(current_key)
    collisions.append(collision_info)
    return keys, collisions


def is_from_maj_voting_part(i, col_idx, pred_col_idx):
    return not ((col_idx - pred_col_idx + 1) <= i <= col_idx)


def sum_secret_distr(distr, weight):
    B = (len(distr) - 1) // 2
    d = defaultdict(float)
    for values in it.product(range(-B, B + 1), repeat=weight):
        d[sum(values)] += prod(distr[val] for val in values)
    return d


base_data_folder = "conditional probs"
prob_filename = os.path.join(base_data_folder, "private_key_and_collision_info.bin")
outfile = open("outfile.txt", "wt")
filename_pattern = os.path.join(
    base_data_folder, "For NO_TESTS is {} alpha_u_and_conditional_probabilities.bin"
)

keys, collisions = parse_key_info_file(prob_filename)

keys_to_test = range(0, 100)

iterations = 10000
# number of coefficients of f
p = 761
# weight of f
w = 286
check_weight = 4

# determine the prior distribution of coefficients of f
f_zero_prob = (p - w) / p
f_one_prob = (1 - f_zero_prob) / 2

f_distr = {-1: f_one_prob, 0: f_zero_prob, 1: f_one_prob}
prior_distr = list(list(sum_secret_distr(f_distr, i + 1).values()) for i in range(2))

differences_arr = []
maj_voting_part_errors_arr = []
non_maj_voting_part_errors_arr = []
full_recovered_keys = 0
for key_idx in keys_to_test:
    if len(collisions[key_idx]) > 1:
        print(f"skipping multiple collision case for {key_idx}")
        continue
    # read posterior distribution of check variables
    filename = filename_pattern.format(key_idx)
    (
        H,
        variable_in_check_idxs,
        check_variables,
        single_check_idxs,
        single_check_distr,
    ) = process_cond_prob_file(filename, p, check_weight)
    col_idx = collisions[key_idx][0][0]

    if H is None or check_variables is None:
        exit()
    row_counts = np.count_nonzero(H, axis=1)
    max_row_weight = np.max(row_counts)
    col_counts = np.count_nonzero(H, axis=0)
    max_col_weight = np.max(col_counts)

    if (max_row_weight - 1) > check_weight:
        print(f"skipping too large predicted collision index for {key_idx}")
        continue

    secret_variables = []

    single_checks = sorted(zip(single_check_idxs, single_check_distr))
    single_checks_idx = 0
    for i in range(p):
        if (
            single_checks_idx < len(single_checks)
            and single_checks[single_checks_idx][0] == i
        ):
            distr = single_checks[single_checks_idx][1]
            secret_variables.append(resize_pmf(distr, B))
            single_checks_idx += 1
        else:
            # If we don't query a single check directly, then usually we just put prior ternary pmf
            # of coefficients of f. However, with extended variables we sometimes need to put pmf
            # of f[i] + f[i+1]
            if i > 0 and i <= col_idx:
                prior_coef_weight = 1
            elif USE_EXTENDED_VARIABLES:
                prior_coef_weight = 2
            else:
                prior_coef_weight = 1
            prior_coef_distr = prior_distr[prior_coef_weight - 1]
            secret_variables.append(resize_pmf(prior_coef_distr, B))

    # convert to numpy arrays for Rust be able to work on the arrays
    secret_variables = np.array(secret_variables, dtype=np.float32)
    check_variables = np.array(check_variables, dtype=np.float32)
    # if collision value is 1, we need to multiply the result by -1
    if collisions[key_idx][0][1] == 1:
        secret_variables = secret_variables[:, ::-1]
        check_variables = check_variables[:, ::-1]

    # Rust does not accept zero values, set them to very small probability
    epsilon = 1e-20
    secret_variables[secret_variables == 0] = epsilon
    check_variables[check_variables == 0] = epsilon

    decoder_map = {
        (False, 2): DecoderNTRUW2,
        (False, 4): DecoderNTRUW4,
        (False, 6): DecoderNTRUW6,
        (True, 2): DecoderExtendedNTRUW2,
        (True, 4): DecoderExtendedNTRUW4,
        (True, 6): DecoderExtendedNTRUW6,
    }
    if USE_EXTENDED_VARIABLES:
        ldpc_check_weight = check_weight // 2
    else:
        ldpc_check_weight = check_weight
    if (USE_EXTENDED_VARIABLES, ldpc_check_weight) not in decoder_map:
        raise ValueError("Not supported check weight")
    decoder = decoder_map[(USE_EXTENDED_VARIABLES, ldpc_check_weight)](
        H.astype("int8"), max_col_weight, max_row_weight, iterations
    )

    f = keys[key_idx]
    s_decoded_pmfs = decoder.decode_with_pr(secret_variables, check_variables)
    fprime = list(np.argmax(pmf) - B for pmf in s_decoded_pmfs)

    for i in range(p):
        if i > 0 and i <= col_idx:
            expect = f[i]
        else:
            expect = f[i] + f[(i - 1) % p]

    # getting from extended representation back to normal
    num_extended = p - col_idx
    matrix = np.zeros((num_extended, p + num_extended), dtype=int)
    for row_idx, i in enumerate(range(col_idx + 1, p + 1)):
        matrix[row_idx, i % p] = 1
        matrix[row_idx, (i - 1) % p] = 1
        matrix[row_idx, p + row_idx] = -1
    row_counts = np.count_nonzero(matrix, axis=1)
    max_row_weight = np.max(row_counts)
    col_counts = np.count_nonzero(matrix, axis=0)
    max_col_weight = np.max(col_counts)

    secret_variables = []
    for i in range(p):
        if i > 0 and i <= col_idx:
            pmf = s_decoded_pmfs[i]
            secret_variables.append(resize_pmf(pmf, 1))
        else:
            secret_variables.append(
                resize_pmf([f_one_prob, f_zero_prob, f_one_prob], 1)
            )
    secret_variables = np.array(secret_variables, dtype=np.float32)
    check_variables = s_decoded_pmfs[col_idx + 1 :] + s_decoded_pmfs[0:1]
    check_variables = np.array(check_variables, dtype=np.float32)
    decoder = decoder_map[(False, 2)](
        matrix.astype("int8"), max_col_weight, max_row_weight, iterations
    )

    secret_variables[secret_variables == 0] = epsilon
    check_variables[check_variables == 0] = epsilon
    s_decoded_pmfs = decoder.decode_with_pr(secret_variables, check_variables)
    # Switching to non-extended representation
    fprime = list(np.argmax(pmf) - 1 for pmf in s_decoded_pmfs)
    differences = 0
    for i in range(p):
        if f[i] != fprime[i]:
            differences += 1
    if differences <= 1:
        full_recovered_keys += 1
    differences_arr.append(differences)

    print(f"For key {key_idx} have total {differences} errors:", file=outfile)

outfile.close()

print(f"Managed to fully recover {full_recovered_keys} keys")
print(f"Average number of errors total is {np.average(differences_arr)}")
