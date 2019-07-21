# ! /anaconda3/envs/metis/bin/ python

"""
Module that scrapes the King County Assessors website for sales data
"""

import pickle
import time
from os import path

import pandas as pd
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver import ActionChains


def open_page(reg_num, sdate, edate):
    """
    Opens a driver from selenium which gets to the selection tab for the region
    from the king county property sales search
    :param reg_num: the alphabetically ordered region number
    :param sdate: start date for the search
    :param edate: end date for the search
    :return: selenium driver
    """
    driver = webdriver.Chrome()
    driver.get('https://info.kingcounty.gov/assessor/esales/Residential.aspx')
    caret = driver.find_element_by_xpath('//*[@id="ResidentialForm"]/ul/li[2]/a/i')
    actionchain = ActionChains(driver)
    actionchain.click(caret).perform()

    time.sleep(1)
    link = driver.find_element_by_xpath('//*[@id="li_Form"]/a')
    actionchain.click(link).perform()

    from_date = driver.find_element_by_id('kcMasterPagePlaceHolder_sf_txtDocDateFrom')
    from_date.send_keys(sdate)

    to_date = driver.find_element_by_id('kcMasterPagePlaceHolder_sf_txtDocDateTo')
    to_date.send_keys(edate)
    caret_2 = driver.find_element_by_xpath(
        '//*[@id="kcMasterPagePlaceHolder_PanelSearch_Res"]/div/div[2]/div/div/'
        'table/tbody/tr[2]/td[1]/div/button/b')
    actionchain.click(caret_2).perform()
    button = driver.find_element_by_xpath(
        f'//*[@id="kcMasterPagePlaceHolder_PanelSearch_Res"]/div/div[2]/div/'
        f'div/table/tbody/tr[2]/td[1]/div/ul/li[{reg_num + 1}]/a/label/input')
    actionchain.click(button).perform()
    return driver


def test(reg_start, reg_end, s_date, e_date):
    """
    Main Scraper function for the king county sale search.
    default starts at region 1, and goes through the whole list
    in practice, you can open many instances of this function to scrape
    multiple pages at once via start and end.

    TODO: implement class
    :param reg_start: number of the region
    :param reg_end: ending bound for scraper
    :param s_date: start date
    :param e_date: end date
    :return: None
    """

    with open('.Region_Names.pickle', 'rb') as f:
        list_of_neighborhoods = pickle.load(f)

    with open('columns.pickle', 'rb') as f:
        column_list = pickle.load(f)

    for i, h_region in enumerate(list_of_neighborhoods):

        if i < reg_start:  # skip to start region
            print('pass', h_region)
            continue
        if i == reg_end:
            return 'Finished'
        if path.exists(f'{h_region}.pickle'):  # catch if already scraped
            continue

        print(f'Starting {list_of_neighborhoods[i]}')

        driver = open_page(i, s_date, e_date)
        actionchain = ActionChains(driver)

        search = driver.find_element_by_id('kcMasterPagePlaceHolder_btnSearch')
        actionchain.click(search).perform()
        search = driver.find_element_by_id('kcMasterPagePlaceHolder_btnSearch')
        num_pages = int(driver.find_element_by_id
                        ('kcMasterPagePlaceHolder_sf_lblDocDateCount').text)
        driver.execute_script("arguments[0].scrollIntoView();", search)
        driver.find_element_by_xpath('//*[@id="kcMasterPagePlaceHolder_btnViewSales"]').click()

        home_df = pd.DataFrame(columns=column_list)

        # king county only shows 5 pages at a time, then a short list for the last page
        full_pages = num_pages // 5
        remainder = num_pages % 5
        idx = 0

        while idx < full_pages:
            try:
                soup = bs(driver.page_source, 'html.parser')
                key = 'kcMasterPagePlaceHolder_gdSalesPagedTransposed'
                home_table = soup.find("table", {"id": key})
                home_list = home_table.find_all('td')
                homes_dict = [[] for _ in range(5)]
                for j in range(len(home_list)):
                    homes_dict[(j % 5)].append(home_list[i].text)
                if idx == 0:
                    home_df = pd.DataFrame(homes_dict)
                else:
                    home_df = home_df.append(pd.DataFrame(homes_dict))

                driver.find_element_by_id('kcMasterPagePlaceHolder_LinkButtonNext').click()
                idx += 1

            except WebDriverException:
                idx += 1

        if remainder != 0:
            try:
                soup = bs(driver.page_source, 'html.parser')
                key = 'kcMasterPagePlaceHolder_gdSalesPagedTransposed'
                home_table = soup.find("table", {"id": key})
                home_list = home_table.find_all('td')
                homes_dict = [[] for _ in range(remainder)]
                for j in range(len(home_list)):
                    homes_dict[(j % remainder)].append(home_list[i].text)
                if idx == 0:
                    home_df = pd.DataFrame(homes_dict)
                else:
                    home_df = home_df.append(pd.DataFrame(homes_dict))

                driver.find_element_by_id('kcMasterPagePlaceHolder_LinkButtonNext').click()
                idx += 1

            except WebDriverException:
                pass

        driver.close()

        if not path.exists(f'{h_region}.pickle'):
            time.sleep(.5)
            with open(f'{h_region}.pickle', 'wb') as f:
                pickle.dump(home_df, f)
        assert path.exists(f'{h_region}.pickle'), f"FAILED TO WRITE {h_region}"
    return None


if __name__ == '__main__':
    test(0, 91, '01/01/2018', '12/31/2018')
