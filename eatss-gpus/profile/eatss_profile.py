#!/usr/bin/env python3
import argparse
import copy
import glob
import math
import trace
import numpy as np
import os
import shutil
import pandas as pd
import re
import subprocess
import time
import traceback
import colorlog
import logging
import logging.config


from contextlib import contextmanager
import signal


from power.power import *
from utils.tile_generator import *


def delete_ppcg_artifacts(target):
    for f in glob.glob("%s*.cu" % target):
        os.remove(f)
    for f in glob.glob("%s*.hu" % target):
        os.remove(f)
    for f in glob.glob("%s*.o" % target):
        os.remove(f)


def make_ppcg(target, path_store, optional_args=[]):
    prog_category, prog_type = lookup_poly_type(
        target, path_store, poly_path='../polybench-c-3.2')
    logger.info("copying Makefile")
    try:
        shutil.copyfile('../Makefile', './Makefile')
    except Exception as e:
        logger.error("Makefile copy failed")
    os.environ["PROG_TARGET"] = target
    os.environ["PROG_CATEGORY"] = prog_category
    if prog_type != '':
        os.environ["PROG_TYPE"] = prog_type
    else:
        os.environ.pop('PROG_TYPE', None)
    res = subprocess.run(["make", "clean"])
    if len(optional_args) == 0:
        res = subprocess.run(["make"], check=True)
    else:
        res = subprocess.run(["make"] + optional_args, check=True)


def optional_lookup_args(target, skip=False, dataset='EXTRALARGE'):
    if skip:
        return [{}]
    if dataset is None:
        cfg = ["-DSTANDARD_DATASET", "-DLARGE_DATASET", "-DEXTRALARGE_DATASET"]
    else:
        cfg = ["-D%s_DATASET" % dataset]
    lookup_ = []
    for c in cfg:
        template = {'ppcg': [c],
                    'make': ["CFLAGS=%s -O3 -lm" % (c)]}
        lookup_.append(template)
    return lookup_


def profiling_subroutine(target, mode, path_store, cmd, column_names, global_df, init, tile_config, skip=False, dataset='EXTRALARGE', power='off', args=[]):
    optional = optional_lookup_args(target, skip, dataset)
    for cfg in optional:
        if len(cfg.keys()) == 0:
            global_df, init = _profiling_subroutine(
                target, mode, path_store, cmd, column_names, global_df, init, tile_config, power, args)
        else:
            ppcg_arg = cfg['ppcg']
            make_arg = cfg['make']
            global_df, init = _profiling_subroutine(
                target, mode, path_store, cmd, column_names, global_df, init, tile_config, power, args, ppcg_arg, make_arg)
    return global_df, init


def _profiling_subroutine(target, mode, path_store, cmd, column_names, global_df, init, tile_config, power, args, ppcg_optional_args=[], optional_args=[]):
    pid = os.getpid()
    artifact_dir = 'artifacts_%d' % pid
    pid = os.getpid()
    artifact_dir = 'artifacts_%d' % pid
    cur_working_dir = os.getcwd()
    try:
        if not os.path.exists(artifact_dir):
            logger.info("creating directory: %s" % artifact_dir)
            os.mkdir(artifact_dir)
            os.makedirs('%s/results' % artifact_dir)
            os.makedirs('%s/logs' % artifact_dir)
    except:
        logger.error("cannot create directory: %s", artifact_dir)
        raise RuntimeError("cannot create directory: %s", artifact_dir)
    os.chdir('./' + artifact_dir)
    delete_ppcg_artifacts(target)
    if len(ppcg_optional_args) > 0:
        cmd.extend(ppcg_optional_args)
    res = subprocess.run(cmd, stdout=subprocess.PIPE)
    if target in ['cholesky', 'adi', 'fdtd-2d', 'jacobi-1d-imper', 'jacobi-2d-imper', 'seidel-2d', 'lu', 'trmm', 'durbin', 'dynprog', 'ludcmp', 'gramschmidt', 'trisolv', 'gemver', 'mvt']:
        iter_lookup = {'cholesky': 1, 'adi': 2, 'fdtd-2d': 2,
                       'jacobi-1d-imper': 1, 'jacobi-2d-imper': 5, 'seidel-2d': 10, 'lu': 1, 'trmm': 1, 'durbin': 1, 'dynprog': 1, 'ludcmp': 1, 'gramschmidt': 1, 'trisolv': 1, 'gemver': 100, 'mvt': 100}
        res = subprocess.run(
            'python ../utils/python_preprocessor.py --benchmark %s --iter %d' % (target, iter_lookup[target]), shell=True)
    else:
        res = subprocess.run(
            'python ../utils/python_preprocessor.py --benchmark %s --iter %d' % (target, args.override_iter), shell=True)
    logger.info("profiling %s with mode %s" % (target, mode))
    try:
        make_ppcg(target, path_store, optional_args)
    except Exception as e:
        os.chdir(cur_working_dir)
        return global_df, init
    finally:
        if not os.path.exists('%s' % target):
            return global_df, init
    _columns = copy.deepcopy(column_names)
    state = True
    if len(ppcg_optional_args) == 0:
        tup = [[target, mode, "None", args.cap_shared_mem, tile_config]]
    else:
        tup = [[target, mode, ' '.join(
            ppcg_optional_args), args.cap_shared_mem, tile_config]]
    if "nvidia-smi" == which_compute():
        state, profiled_info = nvidia_smi_profile(
            target, args.cuda_visible_devices)
    elif "tegra" == which_compute():
        state, profiled_info = tegra_profile(
            target, args.cuda_visible_devices)
    else:
        state, profiled_info = tegra_profile(
            target, args.cuda_visible_devices)
    if state:
        _columns.extend(profiled_info[0])
        tup[0].extend(profiled_info[1:-1])
        tup[0].extend(profiled_info[-1].tolist())
    elif init:
        _columns = global_df.columns
        tup[0] = tup[0] + [-1] * (len(_columns) - len(tup[0]))
    elif not state and not init:
        return global_df, init
    df = pd.DataFrame(tup, columns=_columns)
    if not init:
        global_df = df
        init = True
    else:
        global_df = global_df.append(df)
    os.chdir(cur_working_dir)
    return global_df, init


def get_ppcg_cmd(target, mode, path_store, tile_config_str, override=True, cap_shared_mem=1.0, compute=which_compute(), is_load_from_file=False):
    poly_type, bench = lookup_poly_type(target, path_store)
    fused_command = ["ppcg", "--target=cuda", "--sizes", tile_config_str, "--dump-sizes", "--max-shared-memory", str(get_max_shared_mem_per_sm(cap_shared_mem, compute, is_load_from_file)), "--assume-non-negative-parameters",
                     "--load-schedule", "../custom_schedules/%s-max-fusion-schedule.txt" % (target), "../polybench-c-3.2/%s/%s/%s/%s.c" % (poly_type, bench, target, target), "-I", "../polybench-c-3.2/utilities/"]
    default_command = ["ppcg", "--target=cuda", "--sizes", tile_config_str, "--dump-sizes", "--max-shared-memory", str(get_max_shared_mem_per_sm(cap_shared_mem, compute, is_load_from_file)), "--assume-non-negative-parameters",
                       "../polybench-c-3.2/%s/%s/%s/%s.c" % (poly_type, bench, target, target), "-I", "../polybench-c-3.2/utilities/"]
    if not override:
        fused_command = ["ppcg", "--target=cuda", "--dump-sizes", "--max-shared-memory", str(get_max_shared_mem_per_sm(cap_shared_mem, compute=compute, load_from_file=is_load_from_file)), "--assume-non-negative-parameters",
                         "--load-schedule", "../custom_schedules/%s-max-fusion-schedule.txt" % (target), "../polybench-c-3.2/%s/%s/%s/%s.c" % (poly_type, bench, target, target), "-I", "../polybench-c-3.2/utilities/"]
        default_command = ["ppcg", "--target=cuda", "--dump-sizes", "--max-shared-memory", str(get_max_shared_mem_per_sm(cap_shared_mem, compute=compute, load_from_file=is_load_from_file)), "--assume-non-negative-parameters",
                           "../polybench-c-3.2/%s/%s/%s/%s.c" % (poly_type, bench, target, target), "-I", "../polybench-c-3.2/utilities/"]
    return fused_command if mode == 'max-fusion' else default_command


def lookup_poly_type(benchmark, path_store, poly_path='./polybench-c-3.2'):
    return path_store[benchmark]


def get_benchmark_names(poly_path='./polybench-c-3.2'):
    path_list = []
    for f in glob.glob(poly_path + '/**/*.c', recursive=True):
        path_list.append(f)

    store = {}
    for p in path_list:
        split_ = p.split('/')
        if split_[2] != 'utilities':
            if len(split_) == 5:
                store[split_[3]] = (split_[2], '')
            else:
                store[split_[4]] = (split_[2], split_[3])
    return store


@contextmanager
def timeout(duration):
    def timeout_handler(signum, frame):
        raise Exception(f'block timed out after {duration} seconds')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def explore_space(target, mode, path_store, args):
    cfg = find_base_tile_sizes(
        target, mode, path_store)
    target_cfg = copy.deepcopy(cfg)
    num_dims_dict = {}
    num_dim_lookup = {}
    if args.dataset == 'none':
        dataset = dataset_lookup(which_compute(), target)
    else:
        dataset = args.dataset

    _scaling_factor = get_scaling_factor(which_compute())
    platform = get_platform_name(which_compute())

    if args.tile_strategy == 'load_from_file':
        out_file_name = 'results/intermediate-%s-%s-%s-%s-%s.csv' % (
            target, mode, args.base_tile, args.warp_frac, args.tile_strategy)
        progress_out_file = 'results/progress-%s-%s-%s.csv' % (
            mode, args.base_tile, args.tile_strategy)
    else:
        out_file_name = 'results/intermediate-%s-%s-%s-%s-cap-shared-%.2f.csv' % (
            target, mode, args.base_tile, args.tile_strategy, args.cap_shared_mem)
        progress_out_file = 'results/progress-%s-%s-%s-cap-shared-%.2f.csv' % (
            mode, args.base_tile, args.tile_strategy, args.cap_shared_mem)

    global_df = pd.DataFrame()
    init = False
    restart = None

    p_df = pd.DataFrame(columns=range(2))
    p_df.columns = ["TARGET", "TILECONFIG"]
    if os.path.exists(progress_out_file):
        try:
            p_df = pd.read_csv(progress_out_file)
            restart = p_df.iloc[-1]["TILECONFIG"]
            logger.debug("restarting tile config for %s at %s" % (target, restart))
            restart = json.loads(restart)
            global_df = pd.read_csv(out_file_name)
            init = True
        except Exception as e:
            logger.warning("error loading last tile configuration from checkpoint. please check error: \"%s\"" % e)
            logger.info("default tile configuration will be loaded")
            global_df = pd.DataFrame()
            init = False
            restart = None

    max_dim_problem_size = 4096
    for kernel_idx, dim_vals in cfg['block'].items():
        num_dims_dict[kernel_idx] = len(dim_vals)
        num_dim_lookup[len(dim_vals)] = get_grid_block_size(
            len(dim_vals), max_dim_problem_size)

    max_num_dims = 0
    for kernel_idx, dim_vals in cfg['tile'].items():
        max_num_dims = max(max_num_dims, len(dim_vals))

    for type_, val_ in cfg.items():
        if type_ in ['grid']:
            for idx_, dim_vals_ in val_.items():
                kernel_num_dims = num_dims_dict[idx_]
                grid_block = num_dim_lookup[kernel_num_dims]
                for list_idx, (dim_idx, dim_val) in enumerate(dim_vals_.items()):
                    target_cfg[type_][idx_][dim_idx] = grid_block[type_][list_idx]

    column_names = ["TARGET", "MODE", "OPTIONAL_ARGS", "CAP", "TILECONFIG",
                    "RUNTIME", "AVGPOWER", "MINPOWER", "MAXPOWER"]

    # profile with default tile configuration
    is_load_from_file = False
    if args.tile_strategy != 'load_from_file':
        global_df, init = profiling_subroutine(target, mode, path_store, get_ppcg_cmd(
            target, mode, path_store, "None", override=False, cap_shared_mem=args.cap_shared_mem, compute=which_compute()), column_names, global_df, init, "None", args.multi_dataset, dataset, args.power, args)
    else:
        is_load_from_file = True
    for tile_config_itr in cfp_tile_range(target, args.base_tile, max_num_dims, args.tile_strategy, restart=restart, express=(args.run_express or args.run_express2), warp_frac=args.warp_frac, platform=platform):
        if args.tile_strategy not in ['load_from_file', 'iterative']:
            tile_config = tile_config_itr
        else:
            args.cap_shared_mem = float(tile_config_itr[0])
            tile_config = tile_config_itr[1]
            global_df, init = profiling_subroutine(target, mode, path_store, get_ppcg_cmd(target, mode, path_store, "None", override=False, cap_shared_mem=args.cap_shared_mem, compute=which_compute(
            ), is_load_from_file=is_load_from_file), column_names, global_df, init, "None", args.multi_dataset, dataset, args.power, args)
        for type_, val_ in cfg.items():
            if type_ in ['grid']:
                for idx_, dim_vals_ in val_.items():
                    kernel_num_dims = num_dims_dict[idx_]
                    grid_block = get_adj_grid_size(
                        target, kernel_num_dims, tile_config, dataset)
                    for list_idx, (dim_idx, dim_val) in enumerate(dim_vals_.items()):
                        target_cfg[type_][idx_][dim_idx] = grid_block[list_idx]
            if type_ in ['block']:
                for idx_, dim_vals_ in val_.items():
                    kernel_num_dims = num_dims_dict[idx_]
                    grid_block = get_thread_adj_block_size(
                        kernel_num_dims, tile_config, _scaling_factor)
                    for list_idx, (dim_idx, dim_val) in enumerate(dim_vals_.items()):
                        target_cfg[type_][idx_][dim_idx] = grid_block[list_idx]
            if type_ in ['tile']:
                for idx_, dim_vals_ in val_.items():
                    for list_idx, (dim_idx, dim_val) in enumerate(dim_vals_.items()):
                        target_cfg[type_][idx_][dim_idx] = tile_config[list_idx]
        out = generate_tile_string(target_cfg)
        out = '{' + out + '}'

        # deploy target
        tile_config_ = get_pretty_print_tile_config(target_cfg)
        global_df, init = profiling_subroutine(target, mode, path_store, get_ppcg_cmd(
            target, mode, path_store, out, cap_shared_mem=args.cap_shared_mem, compute=which_compute(), is_load_from_file=is_load_from_file), column_names, global_df, init, tile_config_, args.multi_dataset, dataset, args.power, args)
        global_df.to_csv(out_file_name, index=False)
        p_df.loc[len(p_df)] = [target, tile_config]
        p_df.to_csv(progress_out_file, index=False)
    return global_df


def explore_bench_size(target, mode, path_store, args, dataset):
    cfg = find_base_tile_sizes(target, mode, path_store)
    target_cfg = copy.deepcopy(cfg)
    num_dims_dict = {}
    num_dim_lookup = {}

    if args.tile_strategy == 'load_from_file':
        out_file_name = 'results/intermediate-%s-%s-%s-%s-%s.csv' % (
            target, mode, args.base_tile, args.warp_frac, args.tile_strategy)
        progress_out_file = 'results/progress-%s-%s-%s.csv' % (
            mode, args.base_tile, args.tile_strategy)
    else:
        out_file_name = 'results/intermediate-%s-%s-%s-%s-cap-shared-%.2f.csv' % (
            target, mode, args.base_tile, args.tile_strategy, args.cap_shared_mem)
        progress_out_file = 'results/progress-%s-%s-%s-cap-shared-%.2f.csv' % (
            mode, args.base_tile, args.tile_strategy, args.cap_shared_mem)

    _scaling_factor = get_scaling_factor(which_compute())

    global_df = pd.DataFrame()
    init = False
    restart = None

    p_df = pd.DataFrame(columns=range(2))
    p_df.columns = ["TARGET", "TILECONFIG"]
    if os.path.exists(progress_out_file):
        p_df = pd.read_csv(progress_out_file)
        try:
            restart = p_df.iloc[-1]["TILECONFIG"]
            logger.info("restarting tile config for %s at %s" % (target, restart))
            restart = restart.rstrip('[').lstrip(']').split(', ')
            global_df = pd.read_csv(out_file_name)
            init = False
        except Exception as e:
            logger.error("error restarting at tile size; error: %s", e)

    max_dim_problem_size = 4096
    for kernel_idx, dim_vals in cfg['block'].items():
        num_dims_dict[kernel_idx] = len(dim_vals)
        num_dim_lookup[len(dim_vals)] = get_grid_block_size(
            len(dim_vals), max_dim_problem_size)

    max_num_dims = 0
    for kernel_idx, dim_vals in cfg['tile'].items():
        max_num_dims = max(max_num_dims, len(dim_vals))

    for type_, val_ in cfg.items():
        if type_ in ['grid']:
            for idx_, dim_vals_ in val_.items():
                kernel_num_dims = num_dims_dict[idx_]
                grid_block = num_dim_lookup[kernel_num_dims]
                for list_idx, (dim_idx, dim_val) in enumerate(dim_vals_.items()):
                    target_cfg[type_][idx_][dim_idx] = grid_block[type_][list_idx]

    column_names = ["TARGET", "MODE", "OPTIONAL_ARGS", "CAP", "TILECONFIG",
                    "RUNTIME", "AVGPOWER", "MINPOWER", "MAXPOWER"]

    # profile with default tile configuration
    global_df, init = profiling_subroutine(target, mode, get_ppcg_cmd(
        target, mode, "None", override=False, cap_shared_mem=args.cap_shared_mem), column_names, global_df, init, "None", args.multi_dataset, dataset, args.power, args)
    for tile_config in cfp_tile_range(target, args.base_tile, max_num_dims, args.tile_strategy, restart=restart):
        for type_, val_ in cfg.items():
            if type_ in ['grid']:
                for idx_, dim_vals_ in val_.items():
                    kernel_num_dims = num_dims_dict[idx_]
                    grid_block = get_adj_grid_size(
                        target, kernel_num_dims, tile_config, dataset)
                    for list_idx, (dim_idx, dim_val) in enumerate(dim_vals_.items()):
                        target_cfg[type_][idx_][dim_idx] = grid_block[list_idx]
            if type_ in ['block']:
                for idx_, dim_vals_ in val_.items():
                    kernel_num_dims = num_dims_dict[idx_]
                    grid_block = get_thread_adj_block_size(
                        kernel_num_dims, tile_config, _scaling_factor)
                    for list_idx, (dim_idx, dim_val) in enumerate(dim_vals_.items()):
                        target_cfg[type_][idx_][dim_idx] = grid_block[list_idx]
            if type_ in ['tile']:
                for idx_, dim_vals_ in val_.items():
                    for list_idx, (dim_idx, dim_val) in enumerate(dim_vals_.items()):
                        target_cfg[type_][idx_][dim_idx] = tile_config[list_idx]
        out = generate_tile_string(target_cfg)
        out = '{' + out + '}'

        # deploy target
        tile_config = get_pretty_print_tile_config(target_cfg)
        global_df, init = profiling_subroutine(target, mode, get_ppcg_cmd(
            target, mode, out, cap_shared_mem=args.cap_shared_mem), column_names, global_df, init, tile_config, args.multi_dataset, dataset, args.power, args)
        global_df.to_csv(out_file_name, index=False)
        p_df.loc[len(p_df)] = [target, tile_config]
        p_df.to_csv(progress_out_file, index=False)
    return global_df


def main(args):
    global_df = pd.DataFrame()
    init = False
    store = get_benchmark_names()
    benchmark_names = sorted(store.keys())
    mode = 'default'
    benchmark_names = ["2mm", "3mm", "atax", "bicg", "correlation", "covariance", "gemm", "gemver", "gesummv", "symm", "syr2k", "syrk",
                       "fdtd-2d", "adi", "jacobi-1d-imper", "jacobi-2d-imper", "lu", "ludcmp", "mvt", "trisolv", "gramschmidt", "seidel-2d", "fdtd-apml"]
    if args.fast_forward != -1:
        benchmark_names = benchmark_names[args.fast_forward:]
    if args.run_special:
        benchmark_names = ["2mm", "gemm", "gemver",
                           "mvt", "jacobi-2d-imper", "fdtd-2d"]
    if args.run_special and args.single != 'none':
        benchmark_names = [args.single]
    if args.run_express:
        benchmark_names = ["gemm", "gemver", "fdtd-2d"]
    if args.run_express and args.single != 'none':
        benchmark_names = [args.single]
    if args.run_express2:
        benchmark_names = ["mvt", "jacobi-2d-imper", "2mm"]
    if args.run_express2 and args.single != 'none':
        benchmark_names = [args.single]

    if args.tile_strategy == 'load_from_file':
        progress_out_file = 'results/progress-%s-%s-%s-%s.csv' % (
            mode, args.base_tile, args.warp_frac, args.tile_strategy)
        full_file_name = 'results/tile-exploration-runtime-%s-%s' % (
            args.base_tile, args.tile_strategy)
    else:
        progress_out_file = 'results/progress-%s-%s-%s-cap-shared-%.2f.csv' % (
            mode, args.base_tile, args.tile_strategy, args.cap_shared_mem)
        full_file_name = 'results/tile-exploration-runtime-%s-%s-cap-shared-%.2f' % (
            args.base_tile, args.tile_strategy, args.cap_shared_mem)
    benchmark_idx = 0
    if os.path.exists(progress_out_file):
        logger.info("loading progress from checkpoint...")
        try:
            p_df = pd.read_csv(progress_out_file)
            benchmark_ = p_df.iloc[-1]["TARGET"]
            benchmark_idx = benchmark_names.index(benchmark_)
            logger.info("checkpoint detected. resuming benchmark at: %s" % benchmark_)
        except Exception as e:
            logger.warning("error with loading checkpoint; automated resolver in progress")
            benchmark_idx = 0

    for b_idx in range(benchmark_idx, len(benchmark_names)):
        benchmark = benchmark_names[b_idx]
        logger.info("invoking benchmark: %s" % benchmark)
        if args.tile_strategy == 'load_from_file':
            partial_file_name = 'results/intermediate-%s-%s-%s-%s-%s.csv' % (
                benchmark, mode, args.base_tile, args.warp_frac, args.tile_strategy)
        else:
            partial_file_name = 'results/intermediate-%s-%s-%s-%s-cap-shared-%.2f.csv' % (
                benchmark, mode, args.base_tile, args.tile_strategy, args.cap_shared_mem)

        # ppcg timeout
        try:
            df = explore_space(benchmark, mode, store, args)
        except Exception as e:
            logger.error(e)
            traceback.print_exc()
            logger.warning("skipping benchmark: %s" % benchmark)
            continue
        if not init:
            global_df, init = df, True
        else:
            global_df = global_df.append(df)
        global_df.to_csv(partial_file_name, index=False)
    global_df.to_csv(full_file_name + '-%s.csv' %
                     (int(time.time())), index=False)


def profile_sizes(args):
    logger = args.logger
    global_df = pd.DataFrame()
    init = False
    store = get_benchmark_names()
    mode = 'default'
    for benchmark in ["2mm", "3mm", "gemm", "covariance", "correlation", "atax", "bicg", "trmm"]:
        logger.debug("invoking benchmark: %s" % benchmark)
        for dataset in ["STANDARD", "LARGE", "EXTRALARGE"]:
            logger.debug("dataset chosen:", dataset)
            try:
                df = explore_bench_size(benchmark, mode, store, args, dataset)
            except Exception as e:
                logger.error(e)
                logger.warning("skipping benchmark: ", benchmark)
                continue
            if not init:
                global_df, init = df, True
            else:
                global_df = global_df.append(df)
        global_df.to_csv('results/bench-size-intermediate-%s-%s-%s-%s-cap-shared-%s.csv' %
                         (benchmark, mode, args.base_tile, args.tile_strategy), index=False)
    global_df.to_csv('results/bench-size-runtime-%s-%s-cap-shared-%s-%s.csv' %
                     (args.base_tile, args.tile_strategy, args.cap_shared_mem, int(time.time())), index=False)


if __name__ == '__main__':
    global logger 
    logging.config.fileConfig(fname='config.ini', disable_existing_loggers=False)
    logger = colorlog.getLogger("eatss")
    logger.info("initializing eatss profiler")
    parser = argparse.ArgumentParser(description='Polybench Power Profiler')
    parser.add_argument('--bench-sizes', action='store_true')
    parser.add_argument('--base-tile', type=int, default=get_best_tile_size())
    parser.add_argument('--tile-strategy', type=str,
                        default='mult_one_at_a_time')
    parser.add_argument('--cap-shared-mem', type=float, default=1.00)
    parser.add_argument('--run-special', action='store_const',
                        const=True, default=False)
    parser.add_argument('--multi-dataset',
                        action='store_const', const=False, default=True)
    parser.add_argument('--single', type=str, default='none')
    parser.add_argument('--dataset', type=str, default='none')
    parser.add_argument('--power', type=str, default='on')
    parser.add_argument('--run-express', action='store_const',
                        const=True, default=False)
    parser.add_argument('--run-express2', action='store_const',
                        const=True, default=False)
    parser.add_argument('--override-iter', type=int, default=100)
    parser.add_argument('--cuda-visible-devices', type=str, default=str(0))
    parser.add_argument('--fast-forward', type=int, default=-1)
    parser.add_argument('--warp-frac', type=float, default=1.00)
    args = parser.parse_args()

    if args.bench_sizes:
        args.multi_dataset = False
        profile_sizes(args)
    else:
        main(args)
