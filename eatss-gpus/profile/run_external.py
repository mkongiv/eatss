#!/usr/bin/env python3
import argparse
from contextlib import contextmanager
import copy
import glob
import logging
import logging.config
import os
import shutil
import subprocess
import time
import sys
import traceback

import colorlog
import numpy as np
import pandas as pd

from power.power import *
from external.external_tile_generator import *

global logger 
logging.config.fileConfig(fname='config.ini', disable_existing_loggers=False)
logger = colorlog.getLogger("eatss")
logger.info("initializing eatss profiler")

def delete_ppcg_artifacts(target):
    for f in glob.glob("%s*.cu" % target):
        os.remove(f)
    for f in glob.glob("%s*.hu" % target):
        os.remove(f)
    for f in glob.glob("%s*.o" % target):
        os.remove(f)


def make_ppcg(target, optional_args=[]):
    logger.debug("copying Makefile")
    os.environ["PROG_TARGET"] = target
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


def profiling_subroutine(target, mode, tile_config_list, cmd, column_names, global_df, init, tile_config, skip=False, dataset='EXTRALARGE', power='off', args=[]):
    optional = optional_lookup_args(target, skip, dataset)
    for cfg in optional:
        if len(cfg.keys()) == 0:
            global_df, init = _profiling_subroutine(
                target, mode, tile_config_list, cmd, column_names, global_df, init, tile_config, power, args)
        else:
            ppcg_arg = cfg['ppcg']
            make_arg = cfg['make']
            global_df, init = _profiling_subroutine(
                target, mode, tile_config_list, cmd, column_names, global_df, init, tile_config, power, args, ppcg_arg, make_arg)
    return global_df, init


def _profiling_subroutine(target, mode, tile_config_list, cmd, column_names, global_df, init, tile_config, power, args, ppcg_optional_args=[], optional_args=[]):
    cur_working_dir = os.getcwd()
    os.chdir('./external')
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
        make_ppcg(target, optional_args)
    except Exception as e:
        os.chdir(cur_working_dir)
        return global_df, init
    finally:
        if not os.path.exists('%s' % target):
            os.chdir(cur_working_dir)
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
            target + '_ext', args.cuda_visible_devices)
    elif "tegra" == which_compute():
        state, profiled_info = tegra_profile(
            target + '_ext', args.cuda_visible_devices)
    if state:
        _columns.extend(profiled_info[0])
        tup[0].extend(profiled_info[1:-1])
        tup[0].extend(profiled_info[-1].tolist())
    elif init:
        _columns = global_df.columns
        tup[0] = tup[0] + [-1] * (len(_columns) - len(tup[0]))
    elif not state and not init:
        os.chdir(cur_working_dir)
        return global_df, init
    df = pd.DataFrame(tup, columns=_columns)
    if not init:
        global_df = df
        init = True
    else:
        global_df = global_df.append(df)
    out_code_dir = "%s-%s-SM-%.3f" % (target, '-'.join(
        [str(t) for t in tile_config_list]), args.cap_shared_mem)
    try:
        if not os.path.exists(out_code_dir):
            logger.info("creating directory: %s" % out_code_dir)
            os.mkdir(out_code_dir)
    except:
        raise RuntimeError("cannot create directory: %s", out_code_dir)
    if os.path.exists(out_code_dir):
        try:
            shutil.copyfile('%s_host.cu' %
                            target, '%s/%s_host.cu' % (out_code_dir, target))
            shutil.copyfile('%s_kernel.cu' %
                            target, '%s/%s_kernel.cu' % (out_code_dir, target))
            shutil.copyfile('%s_kernel.hu' %
                            target, '%s/%s_kernel.hu' % (out_code_dir, target))
        except Exception as e:
            logger.error("tiled copy failed")
    os.chdir(cur_working_dir)
    return global_df, init


def get_ppcg_cmd(target, mode, tile_config_str, override=True, cap_shared_mem=1.0, compute=which_compute()):
    fused_command = ["ppcg", "--target=cuda", "--sizes", tile_config_str, "--dump-sizes", "--max-shared-memory", str(get_max_shared_mem_per_sm(cap_shared_mem, compute)), "--assume-non-negative-parameters",
                     "--load-schedule", "../custom_schedules/%s-max-fusion-schedule.txt" % (target), "%s/%s.c" % (target, target), "-I", "../polybench-c-3.2/utilities/"]
    default_command = ["ppcg", "--target=cuda", "--sizes", tile_config_str, "--dump-sizes", "--max-shared-memory", str(get_max_shared_mem_per_sm(cap_shared_mem, compute)), "--assume-non-negative-parameters",
                       "%s/%s.c" % (target, target), "-I", "../polybench-c-3.2/utilities/"]
    if not override:
        fused_command = ["ppcg", "--target=cuda", "--dump-sizes", "--max-shared-memory", str(get_max_shared_mem_per_sm(cap_shared_mem, compute=compute)), "--assume-non-negative-parameters",
                         "--load-schedule", "../custom_schedules/%s-max-fusion-schedule.txt" % (target), "%s/%s.c" % (target, target), "-I", "../polybench-c-3.2/utilities/"]
        default_command = ["ppcg", "--target=cuda", "--dump-sizes", "--max-shared-memory", str(get_max_shared_mem_per_sm(cap_shared_mem, compute=compute)), "--assume-non-negative-parameters",
                           "%s/%s.c" % (target, target), "-I", "../polybench-c-3.2/utilities/"]
    return fused_command if mode == 'max-fusion' else default_command


def explore_space(target, mode, args):
    cfg = find_base_tile_sizes(target, mode)
    target_cfg = copy.deepcopy(cfg)
    num_dims_dict = {}
    num_dim_lookup = {}
    dataset = 'STANDARD'

    _scaling_factor = get_scaling_factor(which_compute())
    platform = get_platform_name(which_compute())

    out_file_name = 'results/ext-intermediate-%s-%s-%s-%s.csv' % (
        target, mode, args.base_tile, args.tile_strategy)
    progress_out_file = 'results/progress-%s-%s-%s.csv' % (
        mode, args.base_tile, args.tile_strategy)

    global_df = pd.DataFrame()
    init = False
    restart = None

    p_df = pd.DataFrame(columns=range(2))
    p_df.columns = ["TARGET", "TILECONFIG"]
    if os.path.exists(progress_out_file):
        try:
            p_df = pd.read_csv(progress_out_file)
            restart = p_df.iloc[-1]["TILECONFIG"]
            logger.info("restarting tile config for %s at %s" % (target, restart))
            restart = json.loads(restart)
            global_df = pd.read_csv(out_file_name)
            init = True
        except Exception as e:
            logger.warning(e)
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
    if args.tile_strategy != 'load_from_file':
        global_df, init = profiling_subroutine(target, mode, get_ppcg_cmd(
            target, mode, "None", override=False, cap_shared_mem=args.cap_shared_mem, compute=which_compute()), column_names, global_df, init, "None", args.multi_dataset, dataset, args.power, args)
    for tile_config_itr in cfp_tile_range(target, args.base_tile, max_num_dims, args.tile_strategy, restart=restart, express=(args.run_express or args.run_express2), platform=platform):
        if args.tile_strategy != 'load_from_file':
            tile_config = tile_config_itr
        else:
            args.cap_shared_mem = float(tile_config_itr[0])
            tile_config = tile_config_itr[1]
            global_df, init = profiling_subroutine(target, mode, tile_config, get_ppcg_cmd(target, mode, "None", override=False, cap_shared_mem=args.cap_shared_mem, compute=which_compute(
            )), column_names, global_df, init, "None", args.multi_dataset, dataset, args.power, args)
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
        global_df, init = profiling_subroutine(target, mode, tile_config, get_ppcg_cmd(
            target, mode, out, cap_shared_mem=args.cap_shared_mem, compute=which_compute()), column_names, global_df, init, tile_config_, args.multi_dataset, dataset, args.power, args)
        global_df.to_csv(out_file_name, index=False)
        p_df.loc[len(p_df)] = [target, tile_config]
        p_df.to_csv(progress_out_file, index=False)
    return global_df


def main(args):
    global_df = pd.DataFrame()
    init = False
    mode = 'default'
    benchmark_names = ['mttkrp']

    progress_out_file = 'results/progress-%s-%s-%s.csv' % (
        mode, args.base_tile, args.tile_strategy)
    benchmark_idx = 0
    if os.path.exists(progress_out_file):
        logger.info("loading progress ...")
        try:
            p_df = pd.read_csv(progress_out_file)
            benchmark_ = p_df.iloc[-1]["TARGET"]
            benchmark_idx = benchmark_names.index(benchmark_)
            logger.info("resuming benchmark at: %s" % benchmark_)
        except Exception as e:
            logger.warning(e)
            benchmark_idx = 0

    for b_idx in range(benchmark_idx, len(benchmark_names)):
        benchmark = benchmark_names[b_idx]
        logger.info("invoking benchmark: %s", benchmark)
        # ppcg timeout
        try:
            df = explore_space(benchmark, mode, args)
        except Exception as e:
            logger.error(e)
            logger.warning("skipping benchmark: ", benchmark)
            continue
        if not init:
            global_df, init = df, True
        else:
            global_df = global_df.append(df)
        global_df.to_csv('results/ext-intermediate-%s-%s-%s-%s.csv' %
                         (benchmark, mode, args.base_tile, args.tile_strategy), index=False)
    global_df.to_csv('results/tile-exploration-runtime-%s-%s-%s.csv' %
                     (args.base_tile, args.tile_strategy, int(time.time())), index=False)


if __name__ == '__main__':
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
    args = parser.parse_args()
    main(args)
