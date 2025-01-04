"""
Step4
输入 修改好的标题等级txt文件, 重建层次的xml数据
输出 未标注的xml隐私政策

Step5

人工标注, 按照知识图谱去标注数据, 标注100份
"""

from body_extraction import bodyExtraction
from bs4 import BeautifulSoup
import bs4
import xml.etree.ElementTree as ET
import re
import os

from loguru import logger

logger.add("{time}.log", level="INFO")

PAGES = "pp_pages"
CHECKED = "titles"
ORIGIN = "china_xml"
TARGET = "results"

TAGS = ['div', 'p', 'ul', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'dt', 'li']
TITLES = ['div', 'p', 'h1', 'h2', 'h3', 'h4', 'dt', 'li']

def tagname_convertion(t):
    tt = int(t)
    if tt < 7:
        return f"h{tt}"
    else:
        return "span"


def node_from_text(body, text, tagname=""):
    titles = body.find_all(tagname, text = text)         #尝试查找当前标题节点
    title = None
    if titles:
        title = titles[-1]
    else:
        text = text.replace('(', '\(')
        text = text.replace(')', '\)')
        pattern = re.compile(f".*{text.strip()}.*")
        titles = body.find_all('',text = pattern)
        if titles:
            title = titles[-1]
        else:
            texts = text.split()
            if len(texts) > 1:
                for t in texts[::-1]:
                    if len(t) < 3: continue
                    pattern = re.compile(f".*{t}.*")
                    titles = body.find_all(text=pattern)
                    if titles:
                        title = titles[-1]
                        break
    if title is None:
        raise RuntimeError(f"标题 {text} 未找到")
    while title and title.name not in TITLES:
        title = title.parent
    
    return title

def get_text_nodes(body):
    res = []
    node = body
    while node:
        # if node.name is None:
        #     if len(node.text.strip()) > 0:
        #         res.append(node)
        if node.name is not None and node.name in TAGS:
            flag = True
            for tag in TAGS:
                if node.find(tag) is not None:
                    flag = False
                    break
            
            if flag and len(node.text.strip()) > 0:
                text = node.text
                if len(text.strip()) > 5 or len(re.sub(r"[0-9]+", "", text.replace(".", "")).strip()) > 1: # 过滤只有标号的标签 如 1.
                    res.append(node)
                nxt = node
                while nxt.next and node in nxt.next.parents: nxt = nxt.next
                node = nxt
        else:
            if node.name is None and len(node.text) > 10 and node.parent.name == 'div' and len(node.parent.text) > 10*len(node.text): # 特殊处理, 不用标签分段的
                res.append(node)
            if node.name == "strong":
                if not node.parent or node.parent.name == "div" or node.parent.name not in TAGS: # 特殊处理 使用strong作为标题
                    res.append(node)
        node = node.next
    return res

def xml_construct2(candidate, html):
    sec1 = sec2 = sec3 = sec4 = root = ET.Element('policy')
    title = None
    num = len(candidate)
    body = bodyExtraction(html)
    text_nodes = get_text_nodes(body)
    # for n in text_nodes: print(n.text.strip())
    title_texts = [''.join(c[7].split()) for c in candidate]
    title_idx = [0] * num
    def check_node_text(node, target):
        if ''.join(node.text.split()) == target: return True
        strings = list(node.strings)
        if target in strings: return True
        return False
    start = 0
    if "隐私" in title_texts[0]: # 如果标题含有隐私政策, ..
        title_idx[0] = 0
        start = 1
    jumped = False
    # 正序查找一次
    cur = 0 # 当前文本节点索引
    while cur < len(text_nodes) and not check_node_text(text_nodes[cur], title_texts[start]): cur += 1
    cur1 = cur
    # 倒序查找, 以跳过目录
    cur = len(text_nodes) - 1 # 当前文本节点索引
    while cur > -1 and not check_node_text(text_nodes[cur], title_texts[start]): cur -= 1
    if cur1 != cur: jumped = True
    if cur/len(text_nodes) > 0.9: # 第一个标题必不会在尾部, 以跳过末尾的侧边栏目录(微信)
        cur -= 1
        jumped = True
        while cur > -1 and not check_node_text(text_nodes[cur], title_texts[start]): cur -= 1 
    assert(cur < len(text_nodes) - 1 and cur > -1)
    
    title_idx[start] = cur
    candidate[start][7] = candidate[start][7].strip()
    for i in range(start + 1, num):
        if False: #jumped or 10*i > num: # 10% 的标题双向查找
            nxt = cur
            while nxt < len(text_nodes) and not check_node_text(text_nodes[nxt], title_texts[i]): nxt += 1
        else:
            # 从前向后找
            nxt = cur
            while nxt < len(text_nodes) and not check_node_text(text_nodes[nxt], title_texts[i]): nxt += 1
            nxt1 = nxt

            # 从后向前找
            nxt = len(text_nodes) - 1
            while nxt > cur and not check_node_text(text_nodes[nxt], title_texts[i]): nxt -= 1
            if nxt1 != len(text_nodes) and nxt != cur and nxt != nxt1: jumped = True
            if nxt/len(text_nodes) > 0.9 and i / num < 0.4: # 前一半标题必不会在尾部, 以跳过末尾的侧边栏目录(微信)
                nxt -= 1
                jumped = True
                while nxt > cur and not check_node_text(text_nodes[nxt], title_texts[i]): nxt -= 1
        if nxt == len(text_nodes) and i != num - 1:
            try:
                node = node_from_text(body, candidate[i][7])
                idx = len(text_nodes) - 1
                while idx > -1 and text_nodes[idx] != node: idx -= 1
            except RuntimeError:
                idx = -1
            
            if idx == -1:
                logger.warning(f"{candidate[i][7]} 未找到")
                
            if i > 0 and idx < title_idx[i - 1]:
                logger.warning(f"{candidate[i][7]} 顺序有问题")
                idx = title_idx[i-1]
        else:
            idx = cur = nxt
            if cur < len(text_nodes):
                candidate[i][7] = candidate[i][7].strip() 
        title_idx[i] = idx

    # title_nodes = list(map(node_from_text, [body] * num, title_texts))

    cur = title_idx[start]
    for i in range(num):
        if candidate[i][-1] == 1:         # 创建section和title节点
            sec4 = sec3 = sec2 = sec1 = ET.SubElement(root, 'section')
            title = ET.SubElement(sec1, 'title')
            title.text = candidate[i][7]
            title.set('tagname', tagname_convertion(candidate[i][4]))
            title.set('depth', candidate[i][3])
            title.set('font-size', candidate[i][1])
            title.set('font-weight', candidate[i][2])
            title.set('category', '')
        elif candidate[i][-1] == 2:
            sec4 = sec3 = sec2 = ET.SubElement(sec1, 'section')
            title = ET.SubElement(sec2, 'title')
            title.text = candidate[i][7]
            title.set('tagname', tagname_convertion(candidate[i][4]))
            title.set('depth', candidate[i][3])
            title.set('font-size', candidate[i][1])
            title.set('font-weight', candidate[i][2])
            title.set('category', '')
        elif candidate[i][-1] == 3:
            sec4 = sec3 = ET.SubElement(sec2, 'section')
            title = ET.SubElement(sec3, 'title')
            title.text = candidate[i][7]
            title.set('tagname', tagname_convertion(candidate[i][4]))
            title.set('depth', candidate[i][3])
            title.set('font-size', candidate[i][1])
            title.set('font-weight', candidate[i][2])
            title.set('category', '')
        elif candidate[i][-1] == 4:
            sec4 = ET.SubElement(sec3, 'section')
            title = ET.SubElement(sec4, 'title')
            title.text = candidate[i][7]
            title.set('tagname', tagname_convertion(candidate[i][4]))
            title.set('depth', candidate[i][3])
            title.set('font-size', candidate[i][1])
            title.set('font-weight', candidate[i][2])
            title.set('category', '')
        else:
            sec5 = ET.SubElement(sec4, 'section')
            title = ET.SubElement(sec5, 'title')
            title.text = candidate[i][7]
            title.set('tagname', tagname_convertion(candidate[i][4]))
            title.set('depth', candidate[i][3])
            title.set('font-size', candidate[i][1])
            title.set('font-weight', candidate[i][2])
            title.set('category', '')
        while (cur:= cur + 1) < len(text_nodes):
            if i < num - 1 and cur == title_idx[i+1] + 1:
                cur -= 1
                break
            if i == num - 1 or cur < title_idx[i+1]:
                tag = text_nodes[cur].name if text_nodes[cur].name else 'p'
                subelement = ET.SubElement(title, tag)
                subelement.set('category', '')
                subelement.text = text_nodes[cur].text.strip()
            else:
                break     

    return root
       
def xml_construct(candidate, filename):
    root = ET.Element('policy')
    body = bodyExtraction(filename)
    num = len(candidate)
    current = root
    current_parent = None
    
    nodes = body.find_all(text = candidate[0][7])
    if nodes:
        node = nodes[-1]
        node = node.next
    else:
        node = body
        start_target = ''.join(s.strip() for s in candidate[0][7].split())
        # 找到开始节点
        while node:
            if node.name is not None:
                text = node.text
                text_strip = ''.join(s.strip() for s in text.split())
                if text_strip == start_target:
                    node = node.next
                    break
            node = node.next
      
    for i in range(num):
        if candidate[i][-1] == 1:         # 创建section和title节点
            sec1 = ET.SubElement(root, 'section')
            title = ET.SubElement(sec1, 'title')
            title.text = candidate[i][7]
            title.set('tagname', tagname_convertion(candidate[i][4]))
            title.set('depth', candidate[i][3])
            title.set('font-size', candidate[i][1])
            title.set('font-weight', candidate[i][2])
            title.set('category', '')
        elif candidate[i][-1] == 2:
            sec2 = ET.SubElement(sec1, 'section')
            title = ET.SubElement(sec2, 'title')
            title.text = candidate[i][7]
            title.set('tagname', tagname_convertion(candidate[i][4]))
            title.set('depth', candidate[i][3])
            title.set('font-size', candidate[i][1])
            title.set('font-weight', candidate[i][2])
            title.set('category', '')
        elif candidate[i][-1] == 3:
            if candidate[i-1][-1] == 1:
                current_parent = sec1
            elif candidate[i-1][-1] == 2:
                current_parent = sec2
            elif candidate[i-1][-1] == 4:  # 注意 此处需要异常处理
                try:
                    current_parent = sec2
                except BaseException:
                    current_parent = sec1
            sec3 = ET.SubElement(current_parent, 'section')
            title = ET.SubElement(sec3, 'title')
            title.text = candidate[i][7]
            title.set('tagname', tagname_convertion(candidate[i][4]))
            title.set('depth', candidate[i][3])
            title.set('font-size', candidate[i][1])
            title.set('font-weight', candidate[i][2])
            title.set('category', '')
        elif candidate[i][-1] == 4:
            if candidate[i-1][-1] == 1:
                current_parent = sec1
            elif candidate[i-1][-1] == 2:
                current_parent = sec2
            elif candidate[i-1][-1] == 3:
                current_parent = sec3
            sec4 = ET.SubElement(current_parent, 'section')
            title = ET.SubElement(sec4, 'title')
            title.text = candidate[i][7]
            title.set('tagname', tagname_convertion(candidate[i][4]))
            title.set('depth', candidate[i][3])
            title.set('font-size', candidate[i][1])
            title.set('font-weight', candidate[i][2])
            title.set('category', '')
        
        if i != num - 1:
            nxt_text = candidate[i+1][7]
            nxt_stripped_text = ''.join(s.strip() for s in nxt_text.split())

            while node:
                if node.name is not None:
                    text = node.text
                    text_strip = ''.join(s.strip() for s in text.split())
                    if text_strip == nxt_stripped_text:
                        node = node.next
                        break
                    if len(text_strip) > 2 and node.name in TAGS:
                        count = 0
                        for tag in TAGS:
                            count += len(node.find_all(tag))
                        if count < 2:
                            subelement = ET.SubElement(title, node.name)
                            subelement.set('category', '')
                            subelement.text = text
                            nxt = node
                            while node in nxt.next.parents: nxt = nxt.next
                            node = nxt
                node = node.next
            if node is None: raise RuntimeError("遍历提早结束")
        else:
            while node:
                if node.name is not None:
                    text = node.text
                    text_strip = ''.join(s.strip() for s in text.split())
                    if len(text_strip) > 2 and node.name in TAGS:
                        count = 0
                        for tag in TAGS:
                            count += len(node.find_all(tag))
                        if count < 2:
                            subelement = ET.SubElement(title, node.name)
                            subelement.set('category', '')
                            subelement.text = text
                            nxt = node
                            while nxt.next and node in nxt.next.parents: nxt = nxt.next
                            node = nxt
                node = node.next
    return root


def prettyXml(element, indent, newline, level=0):
    # 判断element是否有子元素
    if element:
        # 如果element的text没有内容
        if element.text == None or element.text.isspace():
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # 此处两行如果把注释去掉，Element的text也会另起一行
    # else:
    # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list

    for subelement in temp:
        # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
        if temp.index(subelement) < (len(temp) - 1):
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
            # 对子元素进行递归操作
        prettyXml(subelement, indent, newline, level=level + 1)
    return


# 读取人工修正后的candidate列表
def read_candidate(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        c = f.readlines()
    res = []
    for i in range(len(c)):
        c[i].strip()
        if c[i]: # 排除空行
            res.append(eval(c[i][c[i].find('['):]))

    return res


# 用于保留 a 标签，有待调整
def get_strings(body):
    strings = []
    des = list(body.descendants)

    for i in range(len(des)):
        if type(des[i]) == bs4.element.NavigableString and des[i-1].name != 'a':
            strings.append(des[i].string.strip())

        elif des[i].name == 'a':
            strings.append(repr(des[i]))

    return strings


def xml_convertion(filename):
    candidate = read_candidate(filename)
    
    basename = os.path.splitext(os.path.basename(filename))[0]
    html = os.path.join(PAGES, basename+'.html')
    target = os.path.join(TARGET, basename + '.xml')

    root = xml_construct2(candidate, html)  # 生成原始的xml文件,返回根元素结点
    prettyXml(root, '\t', '\n')  # 执行美化方法
    tree = ET.ElementTree(root)
    tree.write(target, encoding='utf-8', xml_declaration=True)

    return tree

if __name__=='__main__':
    if not os.path.exists(TARGET):
        os.makedirs(TARGET)
    files = os.listdir(CHECKED)
    for idx, f in enumerate(files):
        logger.info(f"{idx+1} {f}")
        filename = os.path.join(CHECKED, f)
        try:
            xml_convertion(filename)
        except Exception as e:
            logger.error(e)
