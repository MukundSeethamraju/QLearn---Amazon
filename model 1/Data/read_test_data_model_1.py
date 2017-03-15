import openpyxl
import re
wb = openpyxl.load_workbook('test_questions.xlsx')
sheet  = wb.get_sheet_by_name('test_questions')
f = open('test.txt','w')
for i in range((sheet.max_row) - 1):
    question = str(sheet["B"+str(i+2)].value)
    question = re.sub('[?]', '', question)
    f.write( 'q' + ' ' +question + '\n')
f.close()