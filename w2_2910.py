import camelot
import os
import re

formatType="format2"

folderPath="../w2/"+formatType+"/"

#inputFilePath="../input/W2/pdf/"
inputFilePath="../LP_Samples/W2/"
inpfile='James Kinane 2 W2 and bank transaction summary.pdf'
filename, file_extension = os.path.splitext(inpfile)

outputFilePath="../output/W2/"



def remove(list):
    pattern = '[0-9\-\:]'
    list = [re.sub(pattern, '', i) for i in list]
    return list

def removeJunkIds(list):
    list=[" ".join(list).replace("\n","").strip()]
    return list[0][:10]

# Driver code


def SecondTypeFormat(table):
    extracted_data={}
    extracted_data['employee name']= table.data[17][0]
    extracted_data['employer id num'] = table.data[4][0].split('\n')[0]
    extracted_data['employer name']=table.data[8][0].split('\n')[0]

    extracted_data['Wages, Tips and compensations'] = table.data[2][0].split('\n')[0]
    extracted_data['Social Security wages'] = table.data[4][1].split('\n')[1]
    extracted_data['Medicare wages tips'] = table.data[6][1].split('\n')[1]
    return extracted_data


def FirstTypeFormat(tables):
    extracted_data={}
    extracted_data['employer name'] =tables[1].data[12][0]+" "+tables[1].data[12][1]
    removejunk=remove(tables[2].data[8])
    extracted_data['employee name']=" ".join(removejunk).replace("\n"," ").replace("Social Security Number"," ")
    extracted_data['employer id num'] =removeJunkIds(tables[3].data[1])

    #extracted_data['Wages, Tips and compensations'] = tables[2].data[20][0].split('\n')[2]
    extracted_data['Social Security wages'] =         tables[2].data[1][0].split('\n')[0]
    extracted_data['Medicare wages tips']=            tables[2].data[3][0].split('\n')[0]
    return  extracted_data


def ThirdTypeFormat(tables):
    extracted_data={}
    extracted_data['employer name']=tables[1].data[16][0].split("\n")[2]
    extracted_data['employee name']=tables[1].data[27][0].split("\n")[0]
    extracted_data['employer id num']=tables[1].data[19][0].split("\n")[1]

    extracted_data['Wages, Tips and compensations'] =tables[1].data[20][0].split("\n")[0]
    extracted_data['Social Security wages'] =tables[1].data[23][0].split("\n")[0]
    extracted_data['Medicare wages tips'] =tables[1].data[25][0].split("\n")[0]
    return extracted_data


def extractTableData(inpfile,DocType):
    tables = camelot.read_pdf(inpfile,flavor="stream",pages="all",column_tol=13)

    if DocType == "format1":
        extracted_data = FirstTypeFormat(tables)
    if DocType == "format2":
        extracted_data = SecondTypeFormat(tables[0])
    if DocType=="format3":
        extracted_data=ThirdTypeFormat(tables)
    if DocType=="format5":
        extracted_data=ThirdTypeFormat(tables)

    return extracted_data




# for eachinput in glob(inputFilePath+"/*.pdf"):
from os import listdir
from os.path import isfile, join
filenames = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
for eachFile in filenames:
    inputFilePath=folderPath+eachFile
    #inputFilePath=folderPath+"0064O00000jttNLQAY-00P4O00001JjOs8UAF-salvatore_rabito_w2_or_1040_or.PDF"
    print(inputFilePath)
    extracted_data=(extractTableData(inputFilePath,formatType))
    print("-----------------")
    print()
    print()
    print(extracted_data)
    print()
    print()
    print("-----------------")


# df = pd.DataFrame(data=extracted_data, index=["Value"])
# df = (df.T)
#
# df.to_excel(outputFilePath+filename+"_"+DocType+'_output.xls', index=True,index_label="W2 Title")