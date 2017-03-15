import openpyxl
import re
import pandas as pd
wb = openpyxl.load_workbook('train_questions.xlsx')
sheet  = wb.get_sheet_by_name('train_questions')
df = pd.read_csv('train_labels.csv')

f = open('Train_data.txt','w')
for i in range((sheet.max_row) - 1):
    question = str(sheet["B"+str(i+2)].value)
    question = re.sub('[?]', '', question)
    f.write( str(df['Label'][i]) +':'+ str(df['Label'][i]) +' ' + question + '\n')
f.close()