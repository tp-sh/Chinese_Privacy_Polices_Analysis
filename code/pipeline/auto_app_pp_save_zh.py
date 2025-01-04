# -*- coding: utf-8 -*-

"""
Step1: 隐私政策收集
华为商城各分类top25应用隐私政策
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from loguru import logger
import time


path="C:/Program Files (x86)/Google/Chrome/Application/chromedriver.exe" # chromdriver.exe 放的路径
service = Service(path)
chrome_options = Options()
chrome_options.add_argument(r'--user-data-dir=./user_data') # 用户数据路径
driver = webdriver.Chrome(service=service, options = chrome_options)
driver.implicitly_wait(10)
url = 'https://appstore.huawei.com/Apps' # 华为应用商城
save_path = "./pp_pages"

driver.get(url)
time.sleep(2)

classes = ['教育', '新闻阅读', '拍摄美化', '美食', '出行导航', '旅游住宿', '购物比价', '商务', '儿童', '金融理财', '运动健康', '便捷生活', '汽车']
for class_name in classes:
    try:
        driver.find_elements(By.XPATH, "//div[@class='headerContainer']/div/div/span[contains(text(), '应用')]")[0].click()
        time.sleep(1)
        driver.find_elements(By.XPATH, f"//span[contains(text(), '{class_name}')]")[0].click()
        time.sleep(2)
        names = [e.text for e in driver.find_elements(By.XPATH, "//div[@class='item']/div/div/div/div/span[@class='name']")]
        logger.debug(f"找到{len(names)} 项应用: {names}")
        length = len(names)
        idx = 0
        
        while idx < length:
            try:
                logger.info(f"访问第{idx+1}个应用: {names[idx]}")
                driver.find_elements(By.XPATH, f"//span[contains(text(), '{class_name}')]")[0].click()
                time.sleep(3)
                item = driver.find_elements(By.XPATH, f"//span[contains(text(), '{names[idx]}')]")[0]
                item.click()
                time.sleep(2)
                pp = driver.find_element(by=By.XPATH, value="//div[contains(text(), '隐私政策')]//parent::div")
                if pp:  
                    pre_handle = driver.current_window_handle
                    pp.click()
                    logger.info("打开隐私保护政策")
                    time.sleep(2)
                    if len(driver.window_handles) == 2:
                        driver.switch_to.window(driver.window_handles[-1])
                        with open(f"{save_path}/{class_name}_{idx+1}_{names[idx]}.html", "wb") as f:
                            f.write(driver.page_source.encode("utf8", "ignore"))
                        time.sleep(5)
                        driver.close()
                        driver.switch_to.window(pre_handle)
                    else:
                        logger.error(f"隐私政策: {class_name}_{idx+1}_{names[idx]} 打开失败!")
                # driver.get(url)
                driver.back()
                time.sleep(3)
            except:
                driver.get(url)
                time.sleep(3)
                driver.find_elements(By.XPATH, "//div[@class='headerContainer']/div/div/span[contains(text(), '应用')]")[0].click()
                pass
            idx+=1
    except Exception as e:
        logger.exception(e)
driver.quit()

