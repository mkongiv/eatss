def get_size(target, DATASET="STANDARD"):
    if DATASET == "STANDARD" or DATASET == "None":
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
            "fdtd-2d": [50, 1000],
            "fdtd-apml": 256,
            "jacobi-1d-imper": [100, 10000],
            "jacobi-2d-imper": [20, 1000],
            "seidel-2d": [20, 1000],
            "covariance": 1000,
            "correlation": 1000,
            "conv-2d": [224, 224, 224, 3, 3],
            "heat-3d": [120],
            "mttkrp": [384]
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
            "fdtd-2d": [50, 2000],
            "fdtd-apml": 512,
            "jacobi-1d-imper": [1000, 100000],
            "jacobi-2d-imper": [20, 2000],
            "seidel-2d": [20, 2000],
            "covariance": 2000,
            "correlation": 2000,
            "conv-2d": [224, 224, 224, 3, 3],
            "heat-3d": [120],
            "mttkrp": [384]
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
            "fdtd-2d": [100, 4000],
            "fdtd-apml": 1000,
            "jacobi-1d-imper": [1000, 100000],
            "jacobi-2d-imper": [100, 4000],
            "seidel-2d": [100, 4000],
            "covariance": 4000,
            "correlation": 4000,
            "conv-2d": [224, 224, 224, 3, 3],
            "heat-3d": [200],
            "mttkrp": [384]
        }
    value = store[target]
    if type(value) is not list:
        return [store[target]]
    else:
        return store[target]


def get_flops(target, DATASET="STANDARD"):
    flop_computation = {"2mm": lambda M : 2 * M * M * M * 2, "gemm": lambda M : 2 * M * M * M,
                        "gemver": lambda N: 10 * N * N + N, "mvt": lambda N: 4 * N * N,
                        "fdtd-2d": lambda T, N: N * N * 6 + 4 * (N - 1) * (N - 1), "jacobi-2d-imper":lambda T, N: 5 * 1 * N * N,
                        "3mm": lambda M : 2 * M * M * M * 3, "atax": lambda M : M * M * 4 + M * 2, "bicg": lambda M : M * M * 2 * 2 + 2 * M,
                        "correlation": lambda M : M * M * 2 + M + M + M * M * 6 + M * M * 3 + M * (M - 1) // 2 * M * 2,
                        "covariance": lambda M: M * M  + M * (M - 1) // 2 * M * 2,
                        "gesummv": lambda M: 4 * M * M + 3 * M, "symm": lambda M: 6 * M * M + 5 * M * (M - 2) * (M - 1) // 2,
                        "syr2k": lambda N: N * N + N * N * N * 3, "syrk": lambda N: N * N + N * N * N,
                        "adi": lambda N: 6 * N * N + N + 6 * N * N + N + (N - 2) * N * 3,
                        "jacobi-1d-imper": lambda T, N: 3 * N, "lu": lambda N: (N - 1) * N // 2,
                        "gramschmidt": lambda N: 3 * N + 4 * ((N - 1) * N) // 2,
                        "ludcmp": lambda N: (N - 1) * N // 2 * (N // 2),
                        "trisolv": lambda N: N, "fdtd-apml": lambda N: 34 * N * N * N, "seidel-2d": lambda T, N: 9 * N * N,
                        "conv-2d": lambda NC, NH, NW, KH, KW: NC * (NH - KH) * (NW - KW) * KH * KW * 2,
                        "heat-3d": lambda N: N * N * N * 12 * 2, "mttkrp": lambda N: N * N * N * N * 3}

    # double to float amounts 2x
    flop_count = 2 * flop_computation[target](*get_size(target, DATASET))
    return flop_count
