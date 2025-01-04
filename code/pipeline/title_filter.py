# -*- coding: utf-8 -*-

"""Step3
输入主题提取后的xml, 进行标题过滤, 使用某个分类器, 分类标题等级
输出txt 标题
但是效果不好, 需要人工删除非标题并更正标题等级
"""
from bs4 import BeautifulSoup
import bs4
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException

import csv
from lxml import etree
import lxml.html
from shutil import copyfile
import joblib
import os

from loguru import logger

logger.add("{time}.log", level='DEBUG')

def tdepth(node):
    d = 0
    while node is not None:
        d += 1
        node = node.getparent()
    return d


def tagname_convertion(t):
    if t == 'h1':
        t = 1
    elif t == 'h2':
        t = 2
    elif t == 'h3':
        t = 3
    elif t == 'h4':
        t = 4
    elif t == 'h5':
        t = 5
    elif t == 'h6':
        t = 6
    elif t == 'strong' or t=='b' or t=='span' or t=='em' or t=='u' or t=='i':
        t = 7
    else:
        t = 8
    return t

def leadinglabel(text):
    chineseletter = '一二三四五六七八九十'
    romanletter = ['i','ii','iii','iv','v','vi','vii','viii']
    labelnum = 99
    try:
        labellt = text[0:4]
    except:
        labellt = text
    for i in range(0,len(labellt)):
        for t in range(0,10):
            if labellt[i] == chineseletter[t]:
                labelnum = 1
                try:
                    if labellt[i+1] == '）' or labellt[i+1] == ')':
                        labelnum+=1
                except:
                    continue
        if re.match(r'\d',labellt[i]) != None:
            labelnum = 3
            try:
                if labellt[i+1] == '）'or labellt[i+1] == ')':
                    labelnum+=1
            except:
                continue
        elif re.match(r'[A-Z]',labellt[i]) != None:
            labelnum = 5
            try:
                if labellt[i+1] == '）'or labellt[i+1] == ')':
                    labelnum+=1
            except:
                continue
        elif re.match(r'[a-z]',labellt[i]) != None:
            labelnum = 7
            try:
                if labellt[i+1] == '）'or labellt[i+1] == ')':
                    labelnum+=1
            except:
                continue
        for t in range(0,8):
            if labellt[i] == romanletter[t]:
                labelnum = 9
                try:
                    if labellt[i+1] == '）'or labellt[i+1] == ')':
                        labelnum+=1
                except:
                    continue
    return labelnum

def textlen(t):
    try:
        text_list = t.split()
    except:
        text_list = []
    tlen = len(text_list)
    return tlen
  
# 从简化过的body的descendants中选取候选结点1
def get_candidate(body):
    candidate = []
    tags = list(body.find_all(True))

    for i in range(len(tags)):
        if len(list(tags[i].strings)) == 1:
            candidate.append(tags[i])  # 可能包含text相同的结点
    return candidate        
  
#去掉可能为列表的元素
def label_check(c):
    candidate = []
    set1 = {'td', 'th', 'a'}
    set2 = {'li', 'td', 'th', 'a'}
    for item in c:
        flag = True
        for p in item.parents:
            if p and (p.name in set1):
                flag = False
                break
        if flag and item.name not in set2:
            candidate.append(item)
    return candidate

# 提取每个标签的名称、文本内容 [text,tagname,depth]
def convert_tag_to_list(c, d):
    tags_list = []
    for item in c:
        tag_list = []
        fix_string = item.string.strip().strip('"').strip("'")  # 为了适配Xpath的语法
        tag_list.append(fix_string)
        tag_list.append(item.name)
        depth = len(list(item.parents)) - d
        tag_list.append(depth)

        if tag_list not in tags_list:
            tags_list.append(tag_list)
    return tags_list


# 利用xpath定位元素，再利用selenium获取对应font-size,weight
# tags_list : [text, tagname, depth, font-size, font-weight, color ]
def feature_extract(tags_list, f):
    option = webdriver.ChromeOptions()
    option.add_argument('headless')  # 设置option
    service = Service('C:/Program Files (x86)/Google/Chrome/Application/chromedriver.exe')
    driver = webdriver.Chrome(service=service, options=option)
    # driver.implicitly_wait(6)
    if (not os.path.isabs(f)):  # 需要绝对路径
        f = os.path.abspath(f)
    driver.get("file:///"+f)
    
    html = driver.find_element(By.TAG_NAME, 'html')
    html_fs = html.value_of_css_property('font-size')
    html_fw = html.value_of_css_property('font-weight')
    feature_list = []
        
    for item in tags_list:
        t_text = item[0]
        tagname = item[1]
        depth = item[2]
        tlen = textlen(t_text)
        # print(t_text, tagname, depth)
        logger.info(f"内容: {t_text[:10]} length: {len(t_text)} tag: {tagname} depth:{depth}")
        if tlen != 0:
            labelnum = leadinglabel(t_text)
        else:
            labelnum = 99
        taglevel = tagname_convertion(tagname)
        # 注意Xpath的语法格式, 单双引号的异常处理
        try:
            part_text = [t[:5] for t in re.split('[【】，。：, .?？！=^&*)(、（）a-zA-Z0-9:/\[\]\n]', t_text) if len(t)]
            if len(part_text):
                e = driver.find_element(By.XPATH, f"//*[name()='{tagname}' and contains(string(),'{part_text[0]}')]")
            else:
                logger.warning(f"{t_text} subfind failed")
                continue
        except NoSuchElementException:
            logger.warning("no such Element")
            continue
        except BaseException:
            logger.warning("Base exception")
            continue
        fs = e.value_of_css_property('font-size')
        fw = e.value_of_css_property('font-weight')
        if e.value_of_css_property('text-decoration') == 'underline':
            t_underline = 1
        else:
            t_underline = 2
        if e.value_of_css_property('font-style') == 'italic':
            t_italic = 1
        else:
            t_italic = 2
        
        rel_fs = round(float(fs[:-2]) / float(html_fs[:-2]), 2)
        rel_fw = float(fw) / float(html_fw)
        if tlen<18:
            
            node_feature = [int(labelnum),float(rel_fs),float(rel_fw),int(depth),taglevel,t_italic,t_underline,t_text]
            feature_list.append(node_feature)
        


    # 对于文本内容重复的结点，取特征最显著的: size * weight 最大,；
    #### 该方法有待优化
    unique_feature_list = []
    for i in range(len(feature_list)):
        if i == 0:
            unique_feature_list.append(feature_list[i])

        # tags_list[i][0] 文本内容
        elif feature_list[i][-1] == feature_list[i-1][-1]:
            # >=保证保留 后者 ， 可以使得显性标签（eg. h2）在最外层
            if feature_list[i][1] * feature_list[i][2] >= feature_list[i-1][1] * feature_list[i-1][2]:
                unique_feature_list.pop()
                unique_feature_list.append(feature_list[i])
            else:
                continue
        else:
            unique_feature_list.append(feature_list[i])
    

    return unique_feature_list

def style_feature(c, f, d):

    l1 = convert_tag_to_list(c, d)
    l2 = feature_extract(l1, f)
    return l2

def level_classifier(k):
    clf = joblib.load('ETtrees')
    feature_y = []
    for item in k:
        feature_x = item[0:-1]
        y = clf.predict([feature_x])
        item2 = [y[0],item[-1]]
        feature_y.append(item2)
        
    return feature_y

def get_cluster(c):

    
    for item in c:
        s = c.index(item) +1
        item.append(s)
        
    title_dic = {}
    # {特征四元组 ：序号}
    for item in c:
        t = tuple(item[0:6])
        if t not in title_dic.keys():
            title_dic[t] = [item[-1]]
        else:
            title_dic[t].append(item[-1])

    prio_list = []

    # 选取数量最多的六组
    l1 = sorted(list(title_dic.items()), key=lambda x: len(x[1]), reverse=True)
    if len(l1) <= 6:
        l2 = l1
    else:
        l2 = l1[0:6]

    for i in l2:
        prio_list.append(i[0])  # i[0] 特征四元组

    prio_list.sort(key=lambda x:(x[0], 1/x[1], 1/x[2], x[3], x[4], x[5]))

    candidate = []
    for item in c:
        if tuple(item[0:6]) in prio_list:
            level = prio_list.index(tuple(item[0:6])) + 1
            item.append(level)
            candidate.append(item)

    for i in range(len(candidate)):
        for t in range(0,6):
            candidate[i][t] = str(candidate[i][t])
        

    return candidate

def merge_p_tags(body):
    ps = body.find_all('p')
    for p in ps:
        # p.string = ''.join(p.stripped_strings)
        strings = [s.replace('\n', '') for s in p.strings]
        p.string = ''.join(strings)
    return body

def tag_filter(originname, filename, savename):
    with open(filename, 'r', encoding='utf-8') as cont:
        body = BeautifulSoup(cont, "lxml")
        # 获取主体部分的深度，用于计算每个节点的相对深度
    body_depth = len(list(body.parents))
    # body = merge_p_tags(body) # 是否合并p标签
    candidate1 = get_candidate(body)   # 从所有Tag元素中筛选只包含一个文本节点的元素，含文本重复的结点
    candidate2 = label_check(candidate1)
    candidate3 = style_feature(candidate1, originname, body_depth)
    candidate4 = get_cluster(candidate3)
    with open(savename, 'w', encoding='utf-8') as f:
        for item in candidate4:
            f.write('level:'+str(item[-1])+repr(item) + '\n')

    return
# 将candidate6写入中间文件，人工进行核对     

def xml_filter(filename):
    with open(filename, 'r', encoding='utf-8') as cont:
        body = BeautifulSoup(cont, "lxml")
    if len(body.findAll(True)) < 10:
        return False
    return True


if __name__=="__main__":
    origin = 'pp_pages'
    target = 'china_xml'
    save = 'titles'
    if not os.path.exists(save):
        os.makedirs(save)
    res = []
    for idx, f in enumerate(os.listdir(target)):
        try:
            originname = origin + '/'+f[:-3]+'html'
            filename = target+'/'+f
            savename = save+'/'+f[:-3]+'txt'
            print(originname, filename, savename)
            tag_filter(originname, filename, savename)
        except Exception as e:
            logger.exception(e)
            with open("failed.txt", 'a', encoding='utf8') as ff: 
                ff.write(f+'\n')