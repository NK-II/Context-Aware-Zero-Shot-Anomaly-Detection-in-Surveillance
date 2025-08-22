in_path = 'data/captions.txt'
out_path = 'data/normal_captions.txt'

with open(in_path, 'r') as infile, open(out_path, 'w') as outfile:
    for line in infile:
        if '##' in line:
            caption = line.split('##', 1)[1].strip()
            if caption:
                outfile.write(caption + '\n')

print(f'Extracted caption are stored in {out_path}')