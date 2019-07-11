
#! /anaconda3/envs/metis/bin/ python

import requests
import time
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains
from selenium.common.exceptions import WebDriverException
import chromedriver_binary


driver = webdriver.Chrome()
driver.get('https://info.kingcounty.gov/assessor/esales/Residential.aspx')
caret = driver.find_element_by_xpath('//*[@id="ResidentialForm"]/ul/li[2]/a/i')
actionChain = ActionChains(driver)
actionChain.click(caret).perform()

time.sleep(1)
link = driver.find_element_by_xpath('//*[@id="li_Form"]/a')
actionChain.click(link).perform()

from_date = driver.find_element_by_id('kcMasterPagePlaceHolder_sf_txtDocDateFrom')
from_date.send_keys('01/01/2018')

to_date = driver.find_element_by_id('kcMasterPagePlaceHolder_sf_txtDocDateTo')
to_date.send_keys('12/31/2018')
caret_2 = driver.find_element_by_xpath('//*[@id="kcMasterPagePlaceHolder_PanelSearch_Res"]/div/div[2]/div/div/table/tbody/tr[2]/td[1]/div/button/b')
actionChain.click(caret_2).perform()

neighborhoods = driver.find_element_by_xpath('//*[@id="kcMasterPagePlaceHolder_PanelSearch_Res"]/div/div[2]/div/div/table/tbody/tr[2]/td[1]/div/ul').text
list_of_neighborhoods = neighborhoods.split('\n')
j = 1
print(list_of_neighborhoods[j-1])
button = driver.find_element_by_xpath(f'//*[@id="kcMasterPagePlaceHolder_PanelSearch_Res"]/div/div[2]/div/div/table/tbody/tr[2]/td[1]/div/ul/li[{j}]/a/label/input')
actionChain.click(button).perform()
# for i, c in enumerate(list_of_neighborhoods):
#     if i == 0:
#         actionChain.click(button).perform()
#         continue
#     j = i+1
#     button = driver.find_element_by_xpath(f'//*[@id="kcMasterPagePlaceHolder_PanelSearch_Res"]/div/div[2]/div/div/table/tbody/tr[2]/td[1]/div/ul/li[{j}]/a/label/input')
#
#     actionChain.click(caret_2).perform()
#     actionChain.click(button).perform()


search = driver.find_element_by_id('kcMasterPagePlaceHolder_btnSearch')
actionChain.click(search).perform()
time.sleep(5)
driver.find_element_by_xpath('//*[@id="kcMasterPagePlaceHolder_btnViewSales"]').click()
search = driver.find_element_by_id('kcMasterPagePlaceHolder_btnSearch')
driver.execute_script("arguments[0].scrollIntoView();", search)
driver.find_element_by_xpath('//*[@id="kcMasterPagePlaceHolder_btnViewSales"]').click()
while idx < 100
    try:
        driver.find_element_by_id('kcMasterPagePlaceHolder_LinkButtonNext').click()
    except WebDriverException:
        breakpoint()
