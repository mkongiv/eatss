import fractions
import json
import numpy as np
import os
import pandas as pd
import shutil

from collections import defaultdict

from utils.get_flops import get_flops


def plot_eat(data_dir, fig_dir):
    filenames = os.listdir(data_dir)
    filenames = list(filter(lambda s: s.endswith('csv'), filenames))
    results_dir = f'{data_dir}/eat'

    if not os.path.exists(results_dir):
       try:
           os.makedirs(results_dir)
       except IOError:
           raise Exception("cannot create directory: {0:}".format(results_dir))

    for f in filenames:
        if 'load_from_file' in f and 'tile' not in f and 'progress' not in f and 'ext' not in f:
            shutil.copyfile(f'{data_dir}/{f}', f'{results_dir}/{f}')

    for platform in ['xavier']:  # , 'a100']:
        _plot_eat(data_dir, fig_dir, platform)


def _plot_eat(data_dir, fig_dir, platform='a100'):
    file_dir = f'{data_dir}/eat/'
    names = os.listdir(file_dir)
    names = list(filter(lambda s: s.endswith('.csv'), names))

    df_list = []
    for n in names:
        df = pd.read_csv('{0:}/eat/{2:}'.format(data_dir, platform, n))
        df_list.append(df)
    df = pd.concat(df_list) 

    # drop duplicates
    drop_redundant_base_line = df.groupby(['TARGET', 'TILECONFIG', 'CAP'])
    drop_idx = []
    for name_b, group_b in drop_redundant_base_line:
        idx_ = group_b.index
        drop_idx.extend(idx_[1:])
    df = df.drop(drop_idx)

    df = df.sort_values(by=['TARGET', 'CAP'])
    grouped_df = df.groupby(by=['TARGET'])

    benchmarks = df['TARGET'].unique()
    caps = df['CAP'].unique()
    num_benchmarks = len(benchmarks)
    n_caps = len(caps)
    map_idx = {}
    for c_idx, c in enumerate(caps):
        map_idx[c] = c_idx

    bar_group_width = n_caps
    bar_width = bar_group_width / n_caps

    for idx_, flag in enumerate(['flops', 'energy']):
        stats = defaultdict(lambda: dict())
        for idx, (name, group) in enumerate(grouped_df):
            base_df = group[group['TILECONFIG'] == 'None']
            base_runtime = base_df.loc[:, ['RUNTIME', 'CAP', 'AVGPOWER', 'OPTIONAL_ARGS']]
            base_runtime['CAP_IDX'] = base_runtime['CAP'].map(map_idx)
            flop_count = base_runtime.loc[:, 'OPTIONAL_ARGS'].apply(lambda _dataset: get_flops(name, DATASET=_dataset.lstrip('-D').split('_')[0]))
            base_runtime['GFLOPS'] = flop_count.div((1e9) * base_runtime['RUNTIME'])
            base_runtime['ENERGY'] = base_runtime['RUNTIME'] * base_runtime['AVGPOWER']
            base_execution_time = base_runtime[['CAP_IDX', 'RUNTIME', 'AVGPOWER', 'GFLOPS', 'ENERGY']]
            base_execution_time.columns = ['CAP_IDX', 'RUNTIME_BASE', 'AVGPOWER_BASE', 'GFLOPS_BASE', 'ENERGY_BASE']

            eat_df = group[group['TILECONFIG'] != 'None']
            eat_runtime = eat_df.loc[:, ['RUNTIME', 'CAP', 'AVGPOWER', 'TILECONFIG', 'OPTIONAL_ARGS']]
            eat_runtime['CAP_IDX'] = eat_runtime['CAP'].map(map_idx)
            flop_count = eat_runtime.loc[:, 'OPTIONAL_ARGS'].apply(lambda _dataset: get_flops(name, DATASET=_dataset.lstrip('-D').split('_')[0]))
            eat_runtime['GFLOPS'] = flop_count.div((1e9) * eat_runtime['RUNTIME'])
            eat_runtime['ENERGY'] = eat_runtime['RUNTIME'] * eat_runtime['AVGPOWER']
            eat_execution_time = eat_runtime[['CAP_IDX', 'RUNTIME', 'AVGPOWER', 'TILECONFIG', 'GFLOPS', 'ENERGY']]
            eat_execution_time.columns = ['CAP_IDX', 'RUNTIME_EAT', 'AVGPOWER_EAT', 'TILECONFIG', 'GFLOPS_EAT', 'ENERGY_EAT']

            speedup_df = eat_execution_time.merge(base_execution_time, how='left', on='CAP_IDX')
            if flag == 'flops':
                heights = speedup_df['RUNTIME_BASE'].div(speedup_df['RUNTIME_EAT'])
                min_idx = np.argmax(heights.to_numpy())
                stats[name]['BEST_RUNTIME'] = speedup_df['RUNTIME_EAT'].to_numpy()[min_idx]
                eng = speedup_df['RUNTIME_EAT'] * speedup_df['AVGPOWER_EAT'].to_numpy()
                stats[name]['BEST_GFLOPs'] = speedup_df['GFLOPS_EAT'].to_numpy()[min_idx]
                stats[name]['BEST_ENERGY'] = eng[min_idx]
                stats[name]['BEST_AVG_POWER'] = speedup_df['AVGPOWER_EAT'].to_numpy()[min_idx]
                stats[name]['BEST_PPW'] = np.divide(stats[name]['BEST_GFLOPs'], stats[name]['BEST_AVG_POWER'])
                stats[name]['BASE_RUNTIME'] = speedup_df['RUNTIME_BASE'].to_numpy()[min_idx]
                stats[name]['BASE_GFLOPs'] = speedup_df['GFLOPS_BASE'].to_numpy()[min_idx]
                base_eng = speedup_df['RUNTIME_BASE'] * speedup_df['AVGPOWER_BASE'].to_numpy()
                stats[name]['BASE_ENERGY'] = base_eng[min_idx]
                stats[name]['BASE_AVG_POWER'] = speedup_df['AVGPOWER_BASE'].to_numpy()[min_idx]
                stats[name]['BASE_PPW'] = np.divide(stats[name]['BASE_GFLOPs'], stats[name]['BASE_AVG_POWER'])
            else:
                heights = (speedup_df['RUNTIME_EAT'] * speedup_df['AVGPOWER_EAT']).div(
                    speedup_df['RUNTIME_BASE'] * speedup_df['AVGPOWER_BASE'])

        if flag == 'flops':
            out_df = pd.DataFrame(stats)
            out_df.to_csv('{0:}/{1:}_eat_best.csv'.format(data_dir, platform), index=True)
            print(out_df)

    shared_memory_mapping = {'a100': 196608, 'xavier': 131072, '2080ti': 98304}
    legend_labels = []
    for i in range(len(caps)):
        legend_labels.append(r'shared mem: ${0:.0f}B$ (${1:.0f}\%$)'.format(shared_memory_mapping[platform] * caps[i],
                                                                            (caps[i] / caps[-1]) * 100))
