#!/usr/bin/env python3

def extract(line, pattern, rtype, outfname):
    line = line.strip(pattern)
    vals = line.split(' ')
    vals = [rtype(v) for v in vals[:-1]]

    with open(outfname, 'w') as outfile:
        for v in vals:
            if(rtype == float):
                outfile.write('{:e}\n'.format(v))
            else:
                outfile.write('{:d}\n'.format(v))

    outfile.close()


data_dir = '/home/ducbueno/Tools/opm/opm-simulators/build/'
flags = 6 * [False]

with open(data_dir + 'opencl_output.txt') as infile:
    for line in infile:
        if line.startswith('Cnnzs = '):
            extract(line, 'Cnnzs = ', float, '../data/real/Cnnzs.txt')
            flags[0] = True

        elif line.startswith('Dnnzs = '):
            extract(line, 'Dnnzs = ', float, '../data/real/Dnnzs.txt')
            flags[1] = True

        elif line.startswith('Bnnzs = '):
            extract(line, 'Bnnzs = ', float, '../data/real/Bnnzs.txt')
            flags[2] = True

        elif line.startswith('Ccols = '):
            extract(line, 'Ccols = ', int, '../data/real/Ccols.txt')
            flags[3] = True

        elif line.startswith('Bcols = '):
            extract(line, 'Bcols = ', int, '../data/real/Bcols.txt')
            flags[4] = True

        elif line.startswith('val_pointers = '):
            extract(line, 'val_pointers = ', int, '../data/real/val_pointers.txt')
            flags[5] = True

        if all(flags):
            break

infile.close()

count = 1
with open(data_dir + 'dune_output.txt') as infile:
    for line in infile:
        if line.startswith('x = '):
            extract(line, 'x = ', float, '../data/real/x' + str(count) + '.txt')

        elif line.startswith('y (before) = '):
            extract(line, 'y (before) = ', float, '../data/real/y_before' + str(count) + '.txt')

        elif line.startswith('y (after) = '):
            extract(line, 'y (after) = ', float, '../data/real/y_after' + str(count) + '.txt')
            count = count + 1

        if count > 10:
            break

infile.close()
