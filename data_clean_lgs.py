import json
import csv
from itertools import zip_longest

foldername1to9='mintcand0'
foldername10to13='mintcand'
answer_type=[]
Session=[]
user_statement=[]
chatbot_ans=[]
uName=[]
uID=[]
queryTime=[]
skillsfound=[]
fname=[]
filename = "chatbotlogs.log"
for x in range(1,14):
    if x<10:
        infile=foldername1to9+str(x)+'/'+filename
    elif (x>=10 and x<14 ):
        infile = foldername10to13 + str(x) + '/' + filename
    with open(infile) as  logs_Data:
        logs_Data = logs_Data.readlines()
        for i in logs_Data:
            try:
                if (eval(i.split("- INFO -")[1]))['AnswerType']:
                    answer_type.append((eval(i.split("- INFO -")[1]))['AnswerType'])
                    Session.append((eval(i.split("- INFO -")[1]))['userSession'])
                    user_statement.append((eval(i.split("- INFO -")[1]))['statement'])
                    chatbot_ans.append((eval(i.split("- INFO -")[1]))['response'][0])
                    uName.append((eval(i.split("- INFO -")[1]))['uName'])
                    queryTime.append((eval(i.split("- INFO -")[1]))['queryTime'])
                    skillsfound.append((eval(i.split("- INFO -")[1]))['skills found'])
                    fname.append((eval(i.split("- INFO -")[1]))['uFname'])
            except:
                pass

final_list_toCSV = [queryTime,uName,Session,answer_type, user_statement,chatbot_ans,skillsfound,fname]
export_data = zip_longest(*final_list_toCSV, fillvalue = '')
with open('final_chatbotwithSession_21_08_2018.csv', 'w', encoding='utf-8', newline='') as myfile:
      wr = csv.writer(myfile)
      wr.writerow(("queryTime","uName","Session","answer_type","user_statement","chatbot_ans","skillsfound","First Name"))
      wr.writerows(export_data)
myfile.close()

