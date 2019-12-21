from matplotlib.colors import to_hex

f = open('color.txt', 'r')
o = open('color.py', 'w')

o.write('from matplotlib.colors import to_hex\n')
o.write('\n')
o.write('trace_colors = {\n')


names = []

for line in f:
    res = line.split('#', 1)
    name = res[0].strip()
    names.append(name)

color_codes = []

for name in names:
    color_code = to_hex('xkcd:{}'.format(name))
    rgb = (int(color_code[1:3], 16), 
           int(color_code[3:5], 16), 
           int(color_code[5:7], 16))
    color_codes.append(rgb)

for idx, code in enumerate(color_codes):
    o.write('    {}: {},\n'.format(idx, code))

for idx, (name, code) in enumerate(zip(names, color_codes)):
    o.write('    \"{}\": {}, \n'.format(name, code))
    




o.write('}\n')
