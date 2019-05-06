import os,shutil,random

chars = os.listdir("C:/Users/24400/Desktop/oracle/")

i=0

for char in chars:
	oraclePath = "C:/Users/24400/Desktop/oracle/"+char
	os.mkdir("C:/Users/24400/Desktop/testOracle/"+char)
	jinPath = "C:/Users/24400/Desktop/jin/"+char
	os.mkdir("C:/Users/24400/Desktop/testJin/"+char)
	oracleList = os.listdir(oraclePath)
	jinList = os.listdir(jinPath)
	if len(oracleList)>10:
		oracleList = random.sample(oracleList,10)

	if len(jinList)>10:
		jinList = random.sample(jinList,10)

	for oracle in oracleList:
		originPath = "C:/Users/24400/Desktop/oracle/"+char+"/"+oracle
		destPath = "C:/Users/24400/Desktop/testOracle/"+char+"/"+oracle
		shutil.copy(originPath,destPath)

	for jin in jinList:
		originPath = "C:/Users/24400/Desktop/jin/"+char+"/"+jin
		destPath = "C:/Users/24400/Desktop/testJin/"+char+"/"+jin
		shutil.copy(originPath,destPath)

	i+=1
	print(i)

