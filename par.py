
#! /anaconda3/envs/metis/bin/ python

import requests
import time
import pickle
import pandas as pd
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains
from selenium.common.exceptions import WebDriverException
#import chromedriver_binary
from os import path

def test(test_start):

    f = open('Region_Names.pickle', 'rb')
    list_of_neighborhoods = pickle.load(f)
    f.close()

    # driver = webdriver.Chrome()
    # driver.get('https://info.kingcounty.gov/assessor/esales/Residential.aspx')
    # caret = driver.find_element_by_xpath('//*[@id="ResidentialForm"]/ul/li[2]/a/i')
    # actionChain = ActionChains(driver)
    # actionChain.click(caret).perform()
    #
    # time.sleep(1)
    # link = driver.find_element_by_xpath('//*[@id="li_Form"]/a')
    # actionChain.click(link).perform()
    #
    # from_date = driver.find_element_by_id('kcMasterPagePlaceHolder_sf_txtDocDateFrom')
    # from_date.send_keys('01/01/2018')
    #
    # to_date = driver.find_element_by_id('kcMasterPagePlaceHolder_sf_txtDocDateTo')
    # to_date.send_keys('12/31/2018')
    # caret_2 = driver.find_element_by_xpath('//*[@id="kcMasterPagePlaceHolder_PanelSearch_Res"]/div/div[2]/div/div/table/tbody/tr[2]/td[1]/div/button/b')
    # actionChain.click(caret_2).perform()
    #
    # neighborhoods = driver.find_element_by_xpath('//*[@id="kcMasterPagePlaceHolder_PanelSearch_Res"]/div/div[2]/div/div/table/tbody/tr[2]/td[1]/div/ul').text
    # list_of_neighborhoods = neighborhoods.split('\n')

    #kill = 1
    #start = 26

    for i, h_region in enumerate(list_of_neighborhoods):
        # if i < start:
        #     print('pass', h_region)
        #     continue
        print(list_of_neighborhoods[test_start-i])
        driver = open_page(test_start-i)
        actionChain = ActionChains(driver)
        if i == 1:
            return 'wtf'

        search = driver.find_element_by_id('kcMasterPagePlaceHolder_btnSearch')
        actionChain.click(search).perform()
        search = driver.find_element_by_id('kcMasterPagePlaceHolder_btnSearch')
        num_pages = int(driver.find_element_by_id('kcMasterPagePlaceHolder_sf_lblDocDateCount').text)
        driver.execute_script("arguments[0].scrollIntoView();", search)
        driver.find_element_by_xpath('//*[@id="kcMasterPagePlaceHolder_btnViewSales"]').click()

        home_df = pd.DataFrame()
        full_pages = num_pages // 5
        remainder = num_pages % 5
        idx = 0
        while idx < 2:#full_pages:
            try:
                soup = bs(driver.page_source, 'html.parser')
                key = 'kcMasterPagePlaceHolder_gdSalesPagedTransposed'
                home_table = soup.find("table", {"id": key})
                home_list = home_table.find_all('td')
                homes_dict = [[] for _ in range(5)]
                for i in range(len(home_list)):
                    homes_dict[(i % 5)].append(home_list[i].text)
                if idx == 0:
                    home_df = pd.DataFrame(homes_dict)
                else:
                    home_df = home_df.append(pd.DataFrame(homes_dict))
                    #print(home_df.shape)


                driver.find_element_by_id('kcMasterPagePlaceHolder_LinkButtonNext').click()
                idx += 1

            except WebDriverException:
                idx +=1
                pass
        if remainder != 0:
            try:
                soup = bs(driver.page_source, 'html.parser')
                key = 'kcMasterPagePlaceHolder_gdSalesPagedTransposed'
                home_table = soup.find("table", {"id": key})
                home_list = home_table.find_all('td')
                homes_dict = [[] for _ in range(remainder)]
                for i in range(len(home_list)):
                    homes_dict[(i % remainder)].append(home_list[i].text)
                if idx == 0:
                    home_df = pd.DataFrame(homes_dict)
                else:
                    home_df = home_df.append(pd.DataFrame(homes_dict))
                    #print(home_df.shape)

                driver.find_element_by_id('kcMasterPagePlaceHolder_LinkButtonNext').click()
                idx += 1

            except WebDriverException:
                pass
        driver.close()
        while not path.exists(f'{h_region}_{i}_test.pickle'):
            time.sleep(3)
            with open(f'{h_region}_{i}_test.pickle', "wb") as f:
                pickle.dump(home_df, f)
                f.close
        print(path.exists(f'{h_region}_{i}_test.pickle'))


def open_page(reg_num):
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
    button = driver.find_element_by_xpath(f'//*[@id="kcMasterPagePlaceHolder_PanelSearch_Res"]/div/div[2]/div/div/table/tbody/tr[2]/td[1]/div/ul/li[{reg_num + 1}]/a/label/input')
    actionChain.click(button).perform()
    return driver

if __name__ == '__main__':
    from sys import argv as av
    test(int(av[1]))



