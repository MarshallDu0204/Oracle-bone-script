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