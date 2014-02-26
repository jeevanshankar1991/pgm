



filename = sys.argv[1]
fptr = open(filename, 'r')
for line in fptr:
   a, g, ch, bp, hd, hr, cp, eia, ecg = parse(line)
   
