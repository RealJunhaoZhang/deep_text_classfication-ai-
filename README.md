# deep_text_classfication-ai-

##文件说明
    ### 主要文件结构
    1.原始数据存放在data目录下，代码存放在script目录下
    2.数据处理文件（包括word2vec的训练集，和训练好的模型）由于太大没有上传
    ###其它
    1.script/excel2txt.py是用于把excel中数据导入txt中以便于读入。
    2.script/word2vec.py完成了两个步骤：
    （1）对原始txt文件进行分词，分词结果就是训练集；
    （2）训练word2vec模型。
    3./deep_text_classfication-ai-.code-workspace是vscode工作区文件
    4./f.txt_cut.txt这个文件用处很神奇，总之不要动它就好了

##项目日志
    ###2018/11/29 zjh
    1.原始数据处理：
        为了便于读入数据，把“标注数据汇总.xlsx”中6427则新闻（公告）转成了txt文件（'f.txt'），代码见“script/excel2txt.py”，数据处理文件未上传。
    2.分词：
        利用jieba分词，将上述txt文件读入并分词处理：
        ···
        text = fi.read()  # 获取文本内容
        new_text = jieba.cut(text, cut_all=False)  # 精确模式
        ···
        处理后生成新的txt文件（f_cut.txt，该文件未上传），去除标点符号，作为word2vec的训练集。
    3.训练生成word2vec模型
        代码见“script/word2vec.py”，训练好的模型（f.model）未上传。