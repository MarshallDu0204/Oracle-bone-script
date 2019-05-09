with open("C:/Users/24400/Desktop/solution.txt","r") as f:
	infos = f.readlines()
	infoSet = set()
	for info in infos:
		info = info.split(" ")
		temp = info[0]+" "+info[1]
		infoSet.add(temp)
#print(infoSet)

value = {}

for element in infoSet:
	count = 0
	for info in infos:
		info = info.split(" ")
		temp = info[0]+" "+info[1]
		if temp == element:
			count+=1

	value[element] = count

#print(value)

answerValue = {}

for element in infoSet:
	count = 0
	for info in infos:
		info = info.split(" ")
		temp = info[0]+" "+info[1]
		if temp == element:
			if int(info[0])==int(info[1]):
				if int(info[2])==1:
					count+=1
			else:
				if int(info[2])==0:
					count+=1

	answerValue[element] = count

tpValue = {}
for element in infoSet:
	tp = 0
	for info in infos:
		info = info.split(" ")
		temp = info[0]+" "+info[1]
		if temp == element:
			if int(info[0])==int(info[1]):
				if int(info[2])==1:
					tp+=1

	tpValue[element] = tp

fpValue = {}
for element in infoSet:
	fp = 0
	for info in infos:
		info = info.split(" ")
		temp = info[0]+" "+info[1]
		if temp == element:
			if int(info[0])!=int(info[1]):
				if int(info[2])==0:
					fp+=1

	fpValue[element] = fp

fnValue = {}
for element in infoSet:
	fn = 0
	for info in infos:
		info = info.split(" ")
		temp = info[0]+" "+info[1]
		if temp == element:
			if int(info[0])!=int(info[1]):
				if int(info[2])==1:
					fn+=1

	fnValue[element] = fn

tp = 0
for element in tpValue:
	tp+=tpValue[element]

fp = 0
for element in fpValue:
	fp+=fpValue[element]

fn = 0
for element in fnValue:
	fn+=fnValue[element]

print(tp,fp,fn)

precision = tp/(tp+fp)
recall = tp/(tp+fn)

print(precision,recall)



#print(answerValue)

newAnswer = []

for element in answerValue:
	total = value[element]
	num = answerValue[element]
	div = num/total
	newAnswer.append(div)

print(newAnswer)

totalNum = 0

for element in newAnswer:
	totalNum+=element

totalNum = totalNum/len(newAnswer)
print(totalNum)