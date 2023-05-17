import os

folder = 'datasets/TUD'

for folder, _, files in os.walk(folder):
    for file in files:
        if file[:6] == 'to_TPF' and \
                (file[-4:] == '.pkl' or file[-3:] == '.pt'):
            fpath = os.path.join(folder, file)
            print(fpath)
            os.system(f'rm {fpath}')
