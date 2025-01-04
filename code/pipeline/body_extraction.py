"""
Step 2 获取html后, 提取主体, 得到xml
"""

from bs4 import BeautifulSoup
import bs4
import numpy as np
import re
import xml.etree.ElementTree as ET
import os

def extraction(body):
    new_body = body
    subnodes = list(new_body.contents)  # 包含直接子节点的列表
    std_list = []   # 记录每一次迭代 std ，用于自比较
    while True:
        len_list = []
        for item in subnodes:
            if item.string == None:
                len1 = 0
                for t in item.stripped_strings:
                    len1+=len(repr(t))
                len_list.append(len1)
            else:
                len_list.append(len(repr(item.string)))
            

        len_std = np.std(len_list)
        std_list.append(len_std)

        if std_compare(std_list) or len(len_list) == 1:

            new_body = subnodes[len_list.index(max(len_list))]
            try:
                subnodes = list(new_body.contents)  # 递归寻找最小子树
            except:
                return new_body
            # print(new_body.attrs)  # 打印属性 便于观察

        else:
            # print(std_list)  # 标准差计算结果
            return new_body

def std_compare(std_list):
    if len(std_list) == 1:
        return True
    c = std_list.copy()  # 拷贝保护原始std列表
    for i in range(len(c) - 1, 0, -1):
        if c[i] == 0:
            del c[i]
    add = 0
    for i in range(len(c)-1):
        add += c[i]
    if len(c)-1 == 0:
        return True
    # 最新一次的标准差大于之前平均值*0.6，迭代继续
    elif c[-1] > (add/(len(c)-1)) * 0.6:
        return True
    else:
        return False


# 正文提取  前  的预处理，删除非文本标签, 标签内非空，但不是有效文字
def non_text_label_delete(body):
    for s in body(['img', 'video', 'iframe', 'script', 'style', 'picture', 'source',  'nav', 'footer']):
        s.extract()

    # 删除标签中的style属性，因为往往包含图片src
    for t in body.find_all(True):
        if 'style' in t.attrs:
            del t['style']

    return body


# 正文提取   后   的预处理，删除空标签，即标签里不包含内容
def empty_label_delete(body):

    for item in list(body.descendants):
        if type(item) != bs4.element.NavigableString and (len(list(item.strings)) == 0 or item.string == '\n'):
            item.extract()

    # 删除除a标签为的href属性外,其他全部属性,简化DOM树 ; 删除换行符
    del body.attrs
    for t in body.find_all(True):  # find_all只返回所有目标子节点的 列表
        # 含多项文本的tag的string为None，strings为一个列表
        if t.string: # 删除换行符
            t.string.replace('\n', '')

        if t.name != 'a':
            del t.attrs
        else:
            if t.get('href'):
                a_href = t['href']
                t.attrs.clear()
                p = re.compile(r'[http]|[https]')
                if p.match(a_href):
                    t['href'] = a_href
            else:
                del t.attrs

    for t in body.find_all(True):  # 针对特例设计： ['\n',text,'\n']
        l1 = t.strings

        l2 = [x for x in l1 if x != '\n']

        if len(l2) == 1 and type(t.next_element) == bs4.element.NavigableString:
            t.string = l2[0]

    return body


        
def bodyExtraction(filename):  # main
    with open(filename, 'r', encoding='utf-8') as cont:
        soup = BeautifulSoup(cont, "lxml")

    print("原始大小", len(repr(soup.body)))
    text_only_body = non_text_label_delete(soup.body)  # 删除img等包含非有效文本的标签
    print("删除非文本内容后大小", len(repr(text_only_body)))
    main_body = extraction(text_only_body)  # 主体提取
    print("主体部分大小", len(repr(main_body)))
    try:
        simplified_body = empty_label_delete(main_body)   # 空标签删除
    except:
        simplified_body = main_body 
    lenth_k = len(repr(simplified_body))
    print("简化后大小", lenth_k)

    return simplified_body

if __name__=="__main__":
    target = 'pp_pages'
    save = 'china_xml'
    if not os.path.exists(save):
        os.makedirs(save)
    targets = os.listdir(target)
    for idx, f in enumerate(targets):
        filename = target+'/'+f
        savename = save+'/'+f[:-4]+'xml'
        print(idx+1, filename)
        res = bodyExtraction(filename)
        with open(savename, 'w', encoding='utf8') as f:
            try:
                f.write(res.prettify())
            except:
                f.write(repr(res))




