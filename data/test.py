with open('mnist_seven.csv') as f:
	lines = f.readlines()
for i in range(len(lines)):
	if not('0' == (lines[i][2])):
		print lines[i][2]
