import sys
import json
import random

with open(sys.argv[1], 'r', encoding='utf8') as reader, \
	 open(sys.argv[2], 'w', encoding='utf8') as writer :
	lines=reader.readlines()
	random.seed(2025)
	MAXV=len(lines)//100
	num=0
	print(MAXV)
	S=dict()
	while num < MAXV:
		line=random.choice(lines)
		items = line.strip().split('||')
		context = items[0]
		completion = items[1].strip('\n')
		if context in S.keys():continue
		x = {}
		x['context'] = context #+ '||'
		x['completion'] = completion
		S[context]=1
		writer.write(json.dumps(x)+'\n')
		num+=1

