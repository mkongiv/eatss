import argparse
import os
import re


def process(benchmark_name, num_iters):
    if num_iters == 1:
        return
    f = open("%s_host.cu" % benchmark_name, "r")
    contents = f.read()
    idx = 0
    _kernel_count = len(re.findall('kernel\d+ <.*', contents[idx:]))
    kernel_count = _kernel_count
    for_loop_start = "for (int iter = 0; iter < {0:}; ++iter) {{\n".format(
        num_iters)
    for_loop_end = "}\n"
    start = 0
    end = 0
    kernel_list = []
    # handle fdtd-2d
    for_loop_match = re.search(
        'for\s+\(int\s*\w+\d*\s+=\s+\d+;.*\)', contents[idx:])
    loop_insertion_idx = -1
    loop_end_idx = -1
    if for_loop_match is not None:
        loop_insertion_idx = for_loop_match.start()
        forward_bracket = 1
        char_ptr = for_loop_match.end() + 1
        first = True
        while forward_bracket > 0:
            if contents[char_ptr] == '{':
                if not first:
                    forward_bracket += 1
                first = False
            elif contents[char_ptr] == '}':
                forward_bracket -= 1
            char_ptr += 1
        loop_end_idx = char_ptr

    while kernel_count > 0:
        match = re.search('kernel\d+ <.*', contents[idx:])
        start = idx + match.start()
        end = idx + match.end() + 1
        idx = end
        kernel_count -= 1
        kernel_list.append([start, end])
    cuda_check_kernel = []
    kernel_count = _kernel_count
    idx = 0
    while kernel_count > 0:
        match = re.search('cudaCheckKernel\(\);', contents[idx:])
        start = idx + match.start()
        end = idx + match.end() + 1
        idx = end
        kernel_count -= 1
        cuda_check_kernel.append([start, end])
    skip_list = []
    intermediate_buffer = []
    if _kernel_count > 1:
        skip_list.extend(kernel_list[:-1])
        skip_list.extend(cuda_check_kernel[:-1])
        kernel_length = sum([x[1] - x[0] for x in kernel_list[:-1]])
        # dim block search
        kernel_count = 2 * (_kernel_count - 1)
        kernel_block_list = []
        idx = 0
        while kernel_count > 0:
            match = re.search('dim\d+ k\d+_dim.*', contents[idx:])
            start = idx + match.start()
            end = idx + match.end() + 1
            idx = end
            kernel_count -= 1
            intermediate_buffer.extend(contents[start: end + 1])
    skip_list = sorted(skip_list)

    count_spacing = 0
    ch_idx = kernel_list[-1][0] - 2
    ch = contents[ch_idx]
    while ch == ' ':
        count_spacing += 1
        ch_idx -= 1
        ch = contents[ch_idx]

    buffer = []
    if loop_insertion_idx != -1:
        buffer.extend(contents[0:loop_insertion_idx])
        buffer.extend(for_loop_start)
        last_idx = loop_insertion_idx
        for s in skip_list:
            buffer.extend(contents[last_idx:s[0]])
            last_idx = s[1] + 1
        itmd_idx = kernel_list[-1][0]
        buffer.extend(contents[last_idx: itmd_idx])
        buffer.extend(intermediate_buffer)
        buffer.extend([contents[x[0]: x[1] + 1] for x in kernel_list[:-1]])
        buffer.append(' ' * (count_spacing))
        buffer.extend(contents[kernel_list[-1][0]
                      : cuda_check_kernel[-1][-1] + 1])
        buffer.extend(contents[cuda_check_kernel[-1]
                      [-1] + 1: loop_end_idx + 1])
        buffer.extend(for_loop_end)
        buffer.extend(contents[loop_end_idx + 1:])
    else:
        last_idx = 0
        for s in skip_list:
            buffer.extend(contents[last_idx:s[0]])
            last_idx = s[1] + 1
        itmd_idx = kernel_list[-1][0]
        buffer.extend(contents[last_idx: itmd_idx])
        buffer.extend(intermediate_buffer)
        buffer.extend(for_loop_start)
        buffer.extend([contents[x[0]: x[1] + 1] for x in kernel_list[:-1]])
        buffer.append(' ' * (count_spacing))
        buffer.extend(contents[kernel_list[-1][0]
                      : cuda_check_kernel[-1][-1] + 1])
        buffer.extend(for_loop_end)
        buffer.extend(contents[cuda_check_kernel[-1][-1] + 1:])
    modified_content = ''.join(buffer)
    f = open("%s_host_tmp.cu" % benchmark_name, "w")
    f.write(modified_content)
    os.remove("%s_host.cu" % benchmark_name)
    os.rename("%s_host_tmp.cu" % benchmark_name, '%s_host.cu' % benchmark_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark name')
    parser.add_argument('--benchmark', type=str,
                        help='benchmark to be preprocessed')
    parser.add_argument('--iter', type=int,
                        help='number of iterations', default=100)
    args = parser.parse_args()
    process(args.benchmark, args.iter)
