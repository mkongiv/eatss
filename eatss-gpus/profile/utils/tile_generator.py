#!/usr/bin/env python3
import copy
from collections import defaultdict
import logging
import logging.config
import math
import json
import os
import re
import subprocess

import colorlog
import pandas as pd
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np


global logger 
logging.config.fileConfig(fname='config.ini', disable_existing_loggers=False)
logger = colorlog.getLogger("eatss_profile")
logger.debug("initializing eatss power profiler. testing if nvidia-smi is available...")


def extract_param(_str, table, num_kernels):
    type_ = re.search(r'\w+', _str.split('->')[1]).group(0)
    subtable = table[type_]
    for kernel_idx in range(num_kernels):
        sub_str_g = re.search(r'i0\s=\s%s[\s\d\w=]+(?<!\))' % kernel_idx, _str)
        if sub_str_g is None:
            continue
        sub_str = sub_str_g.group(0)
        matches = re.findall(r'o\d+\s=\s\d+', sub_str)
        for m in matches:
            subtable[kernel_idx][re.search(
                r'o(\d+)', m).group(0)] = int(re.search(r'(?<=\=\s)\d+', m).group(0))


def generate_tile_string(table):
    list_ = []
    for type_, sub_table in table.items():
        num_kernels = len(sub_table.keys())
        sub_length = len(sub_table[list(sub_table.keys())[0]].keys())
        str_ = 'kernel[i0] -> %s' % (type_) + '[%s]' % ', '.join(
            ['o%d' % x for x in range(sub_length)]) + ' : '
        num_dims_dependent = {}
        sub_list_dict = defaultdict(lambda: [])
        for idx, values_ in sub_table.items():
            num_dims = len(values_.keys())
            if num_dims not in num_dims_dependent:
                str_ = 'kernel[i0] -> %s' % (type_) + '[%s]' % ', '.join(
                    ['o%d' % x for x in range(num_dims)]) + ' : '
                num_dims_dependent[num_dims] = str_
            sub_list_dict[num_dims].append('(i0 = %d and ' % idx + ' and '.join(
                ['%s = %d' % (k, v) for (k, v) in zip(values_.keys(), values_.values())]) + ')')
        for num_dim_, str__ in num_dims_dependent.items():
            str_ = str__ + ' or '.join(sub_list_dict[num_dim_])
            list_.append(str_)
    return '; '.join(list_)


def find_base_tile_sizes(target, strategy='default', store={}):
    prog_category, prog_type = store[target]
    if prog_type != '':
        cmd = ["ppcg", "--target=cuda", "--dump-sizes",
               "polybench-c-3.2/%s/%s/%s/%s.c" % (prog_category, prog_type, target, target), "-I", "polybench-c-3.2/utilities/"]
    else:
        cmd = ["ppcg", "--target=cuda", "--dump-sizes",
               "polybench-c-3.2/%s/%s/%s.c" % (prog_category, target, target), "-I", "polybench-c-3.2/utilities/"]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    config_str = out.stderr.decode(
        'utf-8').strip('\n').strip('{').strip('}').rstrip(' ').lstrip(' ')
    split_config = config_str.split(';')
    base_config = defaultdict(lambda: defaultdict(lambda: {}))
    num_kernels = len(set(re.findall(r'i\d+\s=\s\d+', config_str)))
    try:
        for s in split_config:
            extract_param(s, base_config, num_kernels)
    except Exception as e:
        raise RuntimeError('cannot extract block, grid and tile dimensions')
    return base_config


def is_jetpack_available():
    res = subprocess.run(["dpkg -l | grep -i \'jetpack\'"],
                         shell=True, stdout=subprocess.PIPE)
    if "Jetpack" in res.stdout.decode('utf-8').strip('\n'):
        return True


def max_problem_size_lookup(target, DATASET="EXTRALARGE"):
    if DATASET == "STANDARD":
        store = {
            "2mm": 1024,
            "3mm": 1024,
            "atax": 4000,
            "bicg": 4000,
            "cholesky": 1024,
            "doitgen": 128,
            "gemm": 1024,
            "gemver": 4000,
            "gesummv": 4000,
            "mvt": 4000,
            "symm": 1024,
            "syr2k": 1024,
            "syrk": 1024,
            "trisolv": 4000,
            "trmm": 1024,
            "durbin": 4000,
            "dynprog": 50,
            "gramschmidt": 512,
            "lu": 1024,
            "ludcmp": 1024,
            "adi": 1024,
            "fdtd-2d": 1000,
            "fdtd-apml": 256,
            "jacobi-1d-imper": 10000,
            "jacobi-2d-imper": 1000,
            "seidel-2d": 1000,
            "covariance": 1000,
            "correlation": 1000
        }
    elif DATASET == "LARGE":
        store = {
            "2mm": 2000,
            "3mm": 2000,
            "atax": 8000,
            "bicg": 8000,
            "cholesky": 2000,
            "doitgen": 256,
            "gemm": 2000,
            "gemver": 8000,
            "gesummv": 8000,
            "mvt": 8000,
            "symm": 2000,
            "syr2k": 2000,
            "syrk": 2000,
            "trisolv": 8000,
            "trmm": 2000,
            "durbin": 8000,
            "dynprog": 500,
            "gramschmidt": 2000,
            "lu": 2000,
            "ludcmp": 2000,
            "adi": 2000,
            "fdtd-2d": 2000,
            "fdtd-apml": 512,
            "jacobi-1d-imper": 100000,
            "jacobi-2d-imper": 2000,
            "seidel-2d": 2000,
            "covariance": 2000,
            "correlation": 2000
        }
    elif DATASET == "EXTRALARGE":
        store = {
            "2mm": 4000,
            "3mm": 4000,
            "atax": 100000,
            "bicg": 100000,
            "cholesky": 4000,
            "doitgen": 1000,
            "gemm": 4000,
            "gemver": 100000,
            "gesummv": 100000,
            "mvt": 100000,
            "symm": 4000,
            "syr2k": 4000,
            "syrk": 4000,
            "trisolv": 100000,
            "trmm": 4000,
            "durbin": 100000,
            "dynprog": 500,
            "gramschmidt": 4000,
            "lu": 4000,
            "ludcmp": 4000,
            "adi": 4000,
            "fdtd-2d": 4000,
            "fdtd-apml": 1000,
            "jacobi-1d-imper": 100000,
            "jacobi-2d-imper": 4000,
            "seidel-2d": 4000,
            "covariance": 4000,
            "correlation": 4000
        }
    return store[target]


def get_scaling_factor(platform):
    if platform == 'nvidia-smi':
        return 0.5
    elif platform == 'mid-tier':
        return 0.5
    elif platform == 'tegra':
        return 0.5


def get_platform_name(platform):
    if platform == 'nvidia-smi':
        return 'a100'
    elif platform == 'mid-tier':
        return '2080ti'
    elif platform == 'tegra':
        return 'xavier'


def dataset_lookup(platform, target):
    if platform == 'nvidia-smi':
        store = {"2mm": "EXTRALARGE",
                 "3mm": "EXTRALARGE",
                 "atax": 'LARGE',
                 "bicg": 'LARGE',
                 "cholesky": 'EXTRALARGE',
                 "doitgen": 'LARGE',
                 "gemm": 'EXTRALARGE',
                 "gemver": 'LARGE',
                 "gesummv": 'LARGE',
                 "mvt": 'LARGE',
                 "symm": 'EXTRALARGE',
                 "syr2k": 'EXTRALARGE',
                 "syrk": 'EXTRALARGE',
                 "trisolv": 'LARGE',
                 "trmm": 'EXTRALARGE',
                 "durbin": 'LARGE',
                 "dynprog": 'EXTRALARGE',
                 "gramschmidt": 'EXTRALARGE',
                 "lu": 'EXTRALARGE',
                 "ludcmp": 'EXTRALARGE',
                 "adi": 'EXTRALARGE',
                 "fdtd-2d": 'EXTRALARGE',
                 "fdtd-apml": 'EXTRALARGE',
                 "jacobi-1d-imper": 'LARGE',
                 "jacobi-2d-imper": 'EXTRALARGE',
                 "seidel-2d": 'EXTRALARGE',
                 "covariance": 'EXTRALARGE',
                 "correlation": 'EXTRALARGE'}
    elif platform == 'mid-tier':
        store = {"2mm": "STANDARD",
                 "3mm": "STANDARD",
                 "atax": 'STANDARD',
                 "bicg": 'STANDARD',
                 "cholesky": 'STANDARD',
                 "doitgen": 'STANDARD',
                 "gemm": 'STANDARD',
                 "gemver": 'STANDARD',
                 "gesummv": 'STANDARD',
                 "mvt": 'STANDARD',
                 "symm": 'STANDARD',
                 "syr2k": 'STANDARD',
                 "syrk": 'STANDARD',
                 "trisolv": 'STANDARD',
                 "trmm": 'STANDARD',
                 "durbin": 'STANDARD',
                 "dynprog": 'STANDARD',
                 "gramschmidt": 'STANDARD',
                 "lu": 'STANDARD',
                 "ludcmp": 'STANDARD',
                 "adi": 'STANDARD',
                 "fdtd-2d": 'STANDARD',
                 "fdtd-apml": 'STANDARD',
                 "jacobi-1d-imper": 'STANDARD',
                 "jacobi-2d-imper": 'STANDARD',
                 "seidel-2d": 'STANDARD',
                 "covariance": 'STANDARD',
                 "correlation": 'STANDARD'}
    elif platform == 'tegra':
        store = {
            "2mm": 'STANDARD',
            "3mm": 'STANDARD',
            "atax": 'STANDARD',
            "bicg": 'STANDARD',
            "cholesky": 'STANDARD',
            "doitgen": 'STANDARD',
            "gemm": 'STANDARD',
            "gemver": 'STANDARD',
            "gesummv": 'STANDARD',
            "mvt": 'STANDARD',
            "symm": 'STANDARD',
            "syr2k": 'STANDARD',
            "syrk": 'STANDARD',
            "trisolv": 'STANDARD',
            "trmm": 'STANDARD',
            "durbin": 'STANDARD',
            "dynprog": 'STANDARD',
            "gramschmidt": 'STANDARD',
            "lu": 'STANDARD',
            "ludcmp": 'STANDARD',
            "adi": 'STANDARD',
            "fdtd-2d": 'STANDARD',
            "fdtd-apml": 'STANDARD',
            "jacobi-1d-imper": 'STANDARD',
            "jacobi-2d-imper": 'STANDARD',
            "seidel-2d": 'STANDARD',
            "covariance": 'STANDARD',
            "correlation": 'STANDARD'}
    return store[target]


def get_grid_block_size(num_dims, max_dim_problem_size):
    device = cuda.Device(0)
    attrs = device.get_attributes()
    max_block_sizes = [attrs[pycuda.driver.device_attribute.MAX_BLOCK_DIM_X],
                       attrs[pycuda.driver.device_attribute.MAX_BLOCK_DIM_Y], attrs[pycuda.driver.device_attribute.MAX_BLOCK_DIM_Z]]
    max_grid_sizes = [attrs[pycuda.driver.device_attribute.MAX_GRID_DIM_X],
                      attrs[pycuda.driver.device_attribute.MAX_GRID_DIM_Y], attrs[pycuda.driver.device_attribute.MAX_GRID_DIM_Z]]
    if is_jetpack_available():
        max_block_sizes = [32, 16, 16]

    grid_sizes = []
    block_sizes = []
    for i in range(num_dims):
        block_sizes.append(max_block_sizes[i])
        grid_sizes.append(max_grid_sizes[i])
    return {'grid': grid_sizes, 'block': block_sizes}


def get_adj_grid_size(target, num_dims, tiles=[], dataset='STANDARD'):
    max_problem_size = max_problem_size_lookup(target, dataset)
    res = []
    for i in range(num_dims):
        tile = tiles[i]
        res.append((max_problem_size + tile - 1) // tile)
    return res

def get_adj_grid_size_with_problem_size(target, num_dims, tiles=[], problem_size=4000):
    max_problem_size = problem_size
    res = []
    for i in range(num_dims):
        tile = tiles[i]
        res.append((max_problem_size + tile - 1) // tile)
    return res


def get_thread_adj_block_size(num_dims, tile_config=[], scaling_factor=0.5):
    _tile_config = copy.deepcopy(tile_config)

    device = cuda.Device(0)
    attrs = device.get_attributes()
    max_block_size = attrs[pycuda.driver.device_attribute.MAX_THREADS_PER_BLOCK]

    # consider only the first two dimensions
    _max_thread_count = max_block_size * scaling_factor
    logger.debug("maximum thread count per block %s", _max_thread_count)

    for i in range(num_dims):
        if i == 0:
            _max_thread_count //= _tile_config[i]
            continue
        # check maximum available threads per block
        if i == 1 and _tile_config[i] > _max_thread_count:
            _tile_config[i] = int(_max_thread_count)
    logger.debug("tile configuration: %s", _tile_config)
    return _tile_config


def next_power_2(n):
    n = n - 1
    while n & n - 1:
        n = n & n - 1
    return n


def get_max_shared_mem_per_sm(cap_shared_mem=1.0, compute='nvidia-smi', load_from_file=False):
    shared_mem_lookup = {'nvidia-smi': 196608,
                         'tegra': 131072, 'mid-tier': 98304}
    # if compute is not None and not load_from_file:
    #     device = cuda.Device(0)
    #     attrs = device.get_attributes()
    #     max_shared_mem = attrs[pycuda.driver.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]
    # elif compute is not None and load_from_file:
    max_shared_mem = shared_mem_lookup[compute]
    # else:
    #     device = cuda.Device(0)
    #     attrs = device.get_attributes()
    #     # int(shared_mem_lookup[compute])
    #     max_shared_mem = attrs[pycuda.driver.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]
    max_shared_mem *= cap_shared_mem
    if cap_shared_mem < 1.0:
        logger.debug("capping shared memory to: %d", int(max_shared_mem))
    else:
        logger.debug("using full shared memory: %d", int(max_shared_mem))
    return int(max_shared_mem)


def get_best_tile_size(cap_shared_mem=1.0, num_buffers=3):
    device = cuda.Device(0)
    attrs = device.get_attributes()
    memory = attrs[pycuda.driver.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK] * cap_shared_mem
    memory_in_elems = memory / 8
    best = int(math.sqrt(memory_in_elems // num_buffers))
    return next_power_2(best)


def tile_range(n_down, n_up, num_dims):
    idx = [n_down] * num_dims

    def get_tile(dim):
        if dim == 0:
            return []
        return [2 ** idx[dim - 1]] + get_tile(dim - 1)

    def test():
        val = False
        for i in range(num_dims):
            val = val or (idx[i] < n_up)
        return val

    while test():
        for i in range(num_dims):
            yield get_tile(num_dims)
            if idx[i] < n_up:
                idx[i] += 1


def cfp_tile_range(target, base, num_dims, strategy='mult_one_at_a_time', restart=None, express=False, warp_frac=1.0, platform='a100'):
    idx = 1
    val = np.asarray([base] * num_dims)
    multiplier = [x for x in np.around(np.geomspace(1, 2 ** 4, 5)).astype(int)]

    if strategy == 'mult_one_at_a_time':
        while idx < len(multiplier):
            mult = multiplier[idx]
            for i in range(num_dims):
                for j in range(num_dims):
                    if j != i:
                        z = np.copy(val)
                        z[i] *= mult
                        z[j] = int(z[j] * 1 / mult)
                        yield z.tolist()
            idx += 1
    elif strategy == 'mult_one_at_a_time_no_up_scale':
        while idx < len(multiplier):
            mult = multiplier[idx]
            for i in range(num_dims):
                z = np.copy(val)
                z[i] = int(z[i] * 1 / mult)
                yield z.tolist()
            idx += 1
    elif strategy == 'mult_one_at_a_time_no_down_scale':
        while idx < len(multiplier):
            mult = multiplier[idx]
            for i in range(num_dims):
                z = np.copy(val)
                z[i] = z[i] * mult
                yield z.tolist()
            idx += 1
    elif strategy == 'mult_two_at_a_time':
        while idx < len(multiplier):
            mult = multiplier[idx]
            for i in range(num_dims):
                for j in range(i + 1, num_dims):
                    if j != i:
                        z = np.copy(val)
                        z[i] *= mult
                        z[j] *= mult
                        if num_dims > 2:
                            for k in range(num_dims):
                                if k != j and k != i:
                                    z[k] /= mult
                                    yield z.tolist()
                        else:
                            yield z.tolist()
            idx += 1
    elif strategy == 'mult_three_at_a_time':
        while idx < len(multiplier):
            mult = multiplier[idx]
            for i in range(num_dims):
                for j in range(i + 1, num_dims):
                    if num_dims >= 3:
                        if j != i:
                            z = np.copy(val)
                            z[i] *= mult
                            z[j] *= mult
                            for k in range(j + 1, num_dims):
                                if k != j and k != i:
                                    z[k] *= mult
                                    yield z.tolist()
            idx += 1
    elif strategy == 'variant_exploration':
        i_idx, j_idx, k_idx = 0, 0, 0
        if num_dims == 3:
            tile_list_ = [16, 20, 32, 50, 64, 100, 128, 200, 256]
            n = len(tile_list_)
            if restart is not None:
                i_idx, j_idx, k_idx = [tile_list_.index(x) for x in restart]
                k_idx = (k_idx + 1) % n
            logger.info("resuming at: %sx%sx%s" % (tile_list_[
                  i_idx], tile_list_[j_idx], tile_list_[k_idx]))
            for i in range(i_idx, n):
                if restart is not None and i == i_idx:
                    for j in range(j_idx, n):
                        if restart is not None and j == j_idx:
                            for k in range(k_idx, n):
                                yield [tile_list_[i], tile_list_[j], tile_list_[k]]
                        else:
                            for k in range(0, n):
                                yield [tile_list_[i], tile_list_[j], tile_list_[k]]
                else:
                    for j in range(0, n):
                        for k in range(0, n):
                            yield [tile_list_[i], tile_list_[j], tile_list_[k]]

        elif num_dims == 2:
            tile_list_ = [2, 4, 8, 10, 12, 16, 20, 24, 32, 48, 50, 60, 64, 96,
                          100, 128, 192, 200, 225, 250, 256, 300, 384, 400, 420, 450, 500, 512]
            n = len(tile_list_)
            if restart is not None:
                i_idx, j_idx = [tile_list_.index(x) for x in restart]
                j_idx = (j_idx + 1) % n
            logger.info("resuming at: %sx%s" % (tile_list_[i_idx], tile_list_[j_idx]))
            for i in range(i_idx, n):
                if restart is not None and i == i_idx:
                    for j in range(j_idx, n):
                        yield [tile_list_[i], tile_list_[j]]
                else:
                    for j in range(0, n):
                        yield [tile_list_[i], tile_list_[j]]
        elif num_dims == 1:
            tile_list_ = [16, 20, 32, 50, 64, 100, 128, 200, 256]
            n = len(tile_list_)
            if restart is not None:
                i_idx = [tile_list_.index(x) for x in restart]
                i_idx = (i_idx + 1) % n
            logger.info("resuming at: %s" % tile_list_[i_idx])
            for i in range(i_idx, n):
                yield [tile_list_[i]]
    elif strategy == 'load_from_file':
        config_dir = './oracle'
        if platform in ['2080ti']:
            df = pd.read_csv('%s/%s-wf%.1f.tiles' % (config_dir,
                             platform, warp_frac), header=None, sep=':')
            df.columns = ['TILE_CONF', 'TARGET',
                          'CAP', 'WARP_FRAC', 'TILE_CONFIG']
        else:
            df = pd.read_csv('%s/%s.tiles' %
                             (config_dir, platform), header=None, sep=':')
            df.columns = ['TILE_CONF', 'TARGET', 'CAP', 'WARP_FRAC', 'TILE_CONFIG']
        df = df[df['TARGET'] == target]
        cap_arr = df['CAP'].to_numpy()
        tile_arr = df['TILE_CONFIG'].to_numpy()
        for i in range(len(cap_arr)):
            yield [cap_arr[i], [int(x) for x in tile_arr[i].split(' ')]]
    elif strategy == 'ppcg_load_from_file':
        config_dir = './oracle'
        if platform in ['2080ti']:
            df = pd.read_csv('%s/%s-wf%.1f.tiles' % (config_dir,
                             platform, warp_frac), header=None, sep=':')
            df.columns = ['TILE_CONF', 'TARGET',
                          'CAP', 'WARP_FRAC', 'TILE_CONFIG']
        else:
            df = pd.read_csv('%s/a100-ppcg-best-tileconfig-kformat.tiles' %
                             config_dir, header=None, sep=':')
            df.columns = ['TILE_CONF', 'TARGET', 'CAP', 'TILE_CONFIG']
        df = df[df['TARGET'] == target]
        cap_arr = df['CAP'].to_numpy()
        tile_arr = df['TILE_CONFIG'].to_numpy()
        for i in range(len(cap_arr)):
            yield [cap_arr[i], [int(x) for x in tile_arr[i].split(' ')]]
    elif strategy == 'iterative':
        config_dir = './oracle/iterative'
        df = pd.read_csv('%s/%s-wf%.1f.tiles' %
                         (config_dir, platform, warp_frac), header=None, sep=':')
        df.columns = ['SUMMARY', 'TARGET', 'ITR',
                      'CAP', 'WARP_FRAC', 'TILE_CONFIG']
        df['TARGET'] = df['TARGET'].str.split('.', expand=True).iloc[:, 0]
        df = df[df['TARGET'] == target]
        df['CAP'] = df['CAP'].str.split(
            '=', expand=True).iloc[:, 1].astype(np.float)
        cap_arr = df['CAP'].to_numpy()
        tile_arr = df['TILE_CONFIG'].to_numpy()
        for i in range(len(cap_arr)):
            yield [cap_arr[i], [int(x) for x in tile_arr[i].split(',')]]

    elif strategy == 'custom':
        i_idx, j_idx, k_idx = 0, 0, 0
        if target in ["2mm", "gemm"]:
            if not express:
                tile_list_ = [4, 8, 10, 16, 20, 32, 50,
                              64, 96, 100, 128, 192, 200, 250, 256, 512]
            else:
                tile_list_ = [16, 20, 32, 50, 64, 100, 128, 200, 256]
            n = len(tile_list_)
            if restart is not None:
                i_idx, j_idx, k_idx = [tile_list_.index(x) for x in restart]
                k_idx = (k_idx + 1) % n
            logger.info("resuming at: %sx%sx%s" % (tile_list_[
                  i_idx], tile_list_[j_idx], tile_list_[k_idx]))
            for i in range(i_idx, n):
                if restart is not None and i == i_idx:
                    for j in range(j_idx, n):
                        if restart is not None and j == j_idx:
                            for k in range(k_idx, n):
                                yield [tile_list_[i], tile_list_[j], tile_list_[k]]
                        else:
                            for k in range(0, n):
                                yield [tile_list_[i], tile_list_[j], tile_list_[k]]
                else:
                    for j in range(0, n):
                        for k in range(0, n):
                            yield [tile_list_[i], tile_list_[j], tile_list_[k]]

        elif target in ["jacobi-2d-imper", "fdtd-2d"]:
            time_tile_ = [1, 2, 4, 5, 10, 20]
            if not express:
                tile_list_ = [4, 8, 10, 16, 20, 32, 50,
                              64, 96, 100, 128, 192, 200, 250, 256]
            else:
                tile_list_ = [16, 20, 32, 50, 64, 100, 128, 200, 256]
            n = len(tile_list_)
            if restart is not None:
                i_idx, j_idx = [tile_list_.index(x) for x in restart]
                j_idx = (j_idx + 1) % n
            logger.info("resuming at: %sx%s" % (tile_list_[
                  i_idx], tile_list_[j_idx]))
            for i in range(i_idx, n):
                if restart is not None and i == i_idx:
                    for j in range(j_idx, n):
                        yield [tile_list_[i], tile_list_[j]]
                else:
                    for j in range(0, n):
                        yield [tile_list_[i], tile_list_[j]]

        elif target in ["gemver", "mvt"]:
            if not express:
                tile_list_ = [4, 8, 10, 16, 20, 32, 50, 64, 96, 100,
                              128, 192, 200, 250, 256, 384, 400, 500, 512]
            else:
                tile_list_ = [16, 20, 32, 50, 64, 100,
                              128, 200, 256, 384, 400, 500, 512]
            n = len(tile_list_)
            if restart is not None:
                i_idx, j_idx = [tile_list_.index(x) for x in restart]
                j_idx = (j_idx + 1) % n
            logger.info("resuming at: %sx%s" % (tile_list_[
                  i_idx], tile_list_[j_idx]))
            for i in range(i_idx, n):
                if restart is not None and i == i_idx:
                    for j in range(j_idx, n):
                        yield [tile_list_[i], tile_list_[j]]
                else:
                    for j in range(0, n):
                        yield [tile_list_[i], tile_list_[j]]


# additional encoding of np numbers
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_pretty_print_tile_config(table):
    return json.dumps(table, indent=2, cls=NpEncoder)


def get_gpu_stats():
    device = cuda.Device(0)
    attrs = device.get_attributes()
    attr_list = ['T_P_W', 'W_P_S', 'T_P_S', 'B_P_S', 'R_P_S',
                 'R_P_B', 'R_P_T', 'T_P_B', 'cores', 'L1', 'L2']

    store = {'a100': {'W_P_S': 64, 'B_P_S': 32, 'R_P_T': 255, 'L1': 196608, 'cores': 6912},
             '2080ti': {'W_P_S': 32, 'B_P_S': 16, 'R_P_T': 255, 'L1': 32768, 'cores': 4352},
             'xavier': {'W_P_S': 64, 'B_P_S': 32, 'R_P_T': 255, 'L1': 65536, 'cores': 512}}  # 192 KB

    p = 'a100'
    value_list = [attrs[pycuda.driver.device_attribute.WARP_SIZE],  # TPW
                  store[p]['W_P_S'],  # WPS
                  # TPS
                  attrs[pycuda.driver.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR],
                  store[p]['B_P_S'],  # BPS
                  # RPS
                  attrs[pycuda.driver.device_attribute.MAX_REGISTERS_PER_MULTIPROCESSOR],
                  attrs[pycuda.driver.device_attribute.MAX_REGISTERS_PER_BLOCK],  # RPB
                  store[p]['R_P_T'],  # RPT
                  attrs[pycuda.driver.device_attribute.MAX_THREADS_PER_BLOCK],  # TPB
                  store[p]['cores'],  # cores
                  store[p]['L1'],  # L1
                  attrs[pycuda.driver.device_attribute.L2_CACHE_SIZE]]  # L2
    df = pd.DataFrame({0: attr_list, 1: value_list})
    df.to_csv('%s.gpu' % p, header=False, index=False, sep=':')
    logger.info(df)


if __name__ == '__main__':
    # get_gpu_stats()
    for t in cfp_tile_range('2mm', 32, 3, strategy='iterative', restart=None, express=False):
        logger.info(t)
