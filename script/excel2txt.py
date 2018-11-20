#encoding=utf-8

#############################################
###      对word2vec进行训练需要语料库       ###
###  将excel单元格中数据转成txt文件便于读入  ###
#############################################

#转换完成，之后不再执行这段代码

import xlrd

fname = "classfied_data.xlsx"
excelbook = xlrd.open_workbook(r'E:\python\Deep_Text_Classfication\data\classfied_data.xlsx')

def getSheet(sh_index):
    try:
        sh = excelbook.sheet_by_index(sh_index)
    except:
        print('no sheet'+sh_index+' in %s',format(fname))
    return sh

#导入excel数据sheet1
sh1 = getSheet(0)

#获取单元格（5，1）的内容
cell_value = sh1.cell_value(5,1)

#获取单元格（1，1）到（rows-1，1）的内容
i = 1
rows = sh1.nrows
#打开要写入的文件
f=open(r"E:\python\Deep_Text_Classfication\script\f.txt","a+",encoding="utf-8")       # 以追加的方式
#写入……
while i<=(rows-1):
    cv = sh1.cell_value(i,1)
    f.write(cv)
    i += 1