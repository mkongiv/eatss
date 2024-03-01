conda activate eatss-profile
export NCU_PROFILER_PATH=""

# profile EATSS tile sizes
python eatss_profile.py --tile-strategy load_from_file --multi-dataset --override-iter 100 --warp-frac 1.00

# explore the search space
python eatss_profile.py --tile-strategy variant_exploration --multi-dataset --cap-shared-mem 1.0 --override-iter 100

# run experiments with capped shared memory
python eatss_profile.py --run-special --tile-strategy custom --cap-shared-mem 1.0 --multi-dataset --dataset EXTRALARGE

# run non-polybench benchmarks
python run_external.py --tile-strategy load_from_file
