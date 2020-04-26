#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
""" Computes covid-19 cases and deaths per mil people.

This script grabs global covid-19 data and computes total number of cases and
deaths per million people for selected countries. Visual output is provided as
bar graphs.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "bgeneto"
__maintainer__ = "bgeneto"
__contact__ = "bgeneto at gmail"
__copyright__ = "Copyright 2020, bgeneto"
__deprecated__ = False
__license__ = "GPLv3"
__status__ = "Development"
__date__ = "2020/04/25"
__version__ = "0.0.1"

import os
import re
import csv
import sys
import json
import logging
import argparse
import configparser

import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime
from collections import defaultdict, OrderedDict
from urllib.request import urlopen, Request


# exit error codes
NOT_CONNECTED = 1
FILE_NOT_FOUND = 2
DOWNLOAD_FAILED = 3
DATE_FMT_ERROR = 4
NO_DATA_AVAIL = 5

# script name and path
SCRIPT_PATH = os.path.dirname(os.path.abspath(sys.argv[0]))
SCRIPT_NAME = os.path.splitext(os.path.basename(sys.argv[0]))[0]

# logger name
LOGGER = logging.getLogger(SCRIPT_NAME)


def internetConnCheck():
    '''
    Simple internet connection checking by using urlopen.
    Returns True (1) on success or False (0) otherwise.
    '''

    LOGGER.info("Checking your internet connection")

    # test your internet connection against the following sites:
    TEST_URL = ("google.com", "search.yahoo.com", "bing.com")

    # quick check using urlopen
    for url in TEST_URL:
        try:
            con = urlopen("http://"+url, timeout=10)
            con.read()
            con.close()
            return True  # no need to not perform additional tests
        except Exception:
            continue

    return False  # all urls failed!


def setupLogging(verbose=False):
    """
    Configure script log system
    """
    ch, fh = None, None
    # starts with the highest logging level available
    numloglevel = logging.DEBUG
    LOGGER.setLevel(numloglevel)
    # setup a console logging first
    log2con = int(getIniSetting('log', 'log_to_stdout'))
    if log2con == 1:
        ch = logging.StreamHandler()
        ch.setLevel(numloglevel)
        formatter = logging.Formatter('%(levelname)8s - %(message)s')
        ch.setFormatter(formatter)
        LOGGER.addHandler(ch)

    # now try set log level according to ini file setting
    try:
        loglevel = getIniSetting('log', 'log_level')
        if not verbose:
            numloglevel = getattr(logging, loglevel.upper(), None)
        LOGGER.setLevel(numloglevel)
        if ch:
            ch.setLevel(numloglevel)
    # just in case (safeguard for corrupted .ini file)
    except configparser.NoSectionError:
        createIniFile()
        loglevel = getIniSetting('log', 'log_level')
        if not verbose:
            numloglevel = getattr(logging, loglevel.upper(), None)
        LOGGER.setLevel(numloglevel)
        ch.setLevel(numloglevel)

    # setup file logging
    log2file = int(getIniSetting('log', 'log_to_file'))
    if log2file == 1:
        logfilename = os.path.join(SCRIPT_PATH, "{}.log".format(SCRIPT_NAME))
        fh = logging.FileHandler(logfilename, mode='w')
        fh.setLevel(numloglevel)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        LOGGER.addHandler(fh)

    return (ch, fh)


def createIniFile(fn=os.path.join(SCRIPT_PATH, f"{SCRIPT_NAME}.ini")):
    '''
    Creates an initial config file with default values
    '''
    config = configparser.ConfigParser()
    config.add_section("url")
    config.set("url", "cases",
               "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
    config.set("url", "deaths",
               "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
    config.set("url", "population",
               "https://www.worldometers.info/world-population/{code}-population/")
    config.set("url", "all",
               "https://www.worldometers.info/coronavirus/country/{code}/")
    config.add_section("log")
    config.set("log", "log_to_file", "1")
    config.set("log", "log_to_stdout", "1")
    config.set("log", "log_level", "info")
    config.add_section("country")
    config.set("country", "country_filename", "countries.txt")

    with open(fn, "w") as configFile:
        try:
            config.write(configFile)
        except Exception:
            LOGGER.critical("Unable to write initial config file")
            LOGGER.critical("Check your file permissions and try again")


def getIniConfig():
    '''
    Returns the config object
    '''
    fn = os.path.join(SCRIPT_PATH, f"{SCRIPT_NAME}.ini")
    if not os.path.isfile(fn):
        createIniFile(fn)

    config = configparser.ConfigParser()
    config.read(fn)
    return config


def getIniSetting(section, setting):
    '''
    Return a setting value
    '''
    config = getIniConfig()
    value = config.get(section, setting)
    return value


def setupCmdLineArgs():
    """ 
    Setup script command line arguments
    """
    parser = argparse.ArgumentParser(
        description='This python script computes confirmed covid-19 total number '
                    'of cases and deaths per million people for selected countries '
                    '(see .ini file) and plots those data as bar charts')
    parser.add_argument('-v', '--version', action='version',
                        version=f'%(prog)s v{__version__}')
    parser.add_argument('-i', '--internet', action='store_true', default=False,
                        help='do NOT check for an active Internet connection')
    parser.add_argument('-f', '--force', action='store_true',
                        default=False, help='force download of new data')
    parser.add_argument('-d', '--day', action='store',
                        dest='day', default=None, help='show data for this day only (YYYYMMDD)')
    parser.add_argument('-c', '--country', action='store',
                        dest='country', default=None, help='show data for this country only (ISO 3166 2-digit code)')
    args = parser.parse_args()

    return args


def mergeDicts(d1, d2):
    """
    Merge two dictionaries 
    """
    merged = defaultdict(list)
    for d in (d1, d2):
        for key, value in d.items():
            if value is not None:
                merged[key].append(value)

    return merged


def getCountries():
    '''
    Read countries from text file
    Returns: list of countries
    '''

    # read country list from input text file
    countries = None
    country_filename = getIniSetting("country", "country_filename")
    if os.path.isfile(os.path.join(SCRIPT_PATH, country_filename)):
        with open(country_filename) as file:
            countries = [line.strip() for line in file]
    else:
        LOGGER.critical(f"File '{country_filename}' not found")
        LOGGER.critical(
            "Please double-check the 'country_filename' ini setting and try again")
        sys.exit(FILE_NOT_FOUND)

    return countries


def checkPopulationFile(countries, pop_filename):
    """
    Check if json population file contains all the countries we need
    Returns: None or dict[country] = population_value
    """
    # check if json population file already exists
    population = []
    if os.path.isfile(pop_filename):
        try:
            with open(pop_filename) as fp:
                population = json.load(fp)
        except:
            LOGGER.error("Error reading from file '{}'".format(
                os.path.basename(pop_filename)))
            return None
    else:
        LOGGER.debug("Population json file '{}' does not exists".format(
            os.path.basename(pop_filename)))
        return None

    # check if we have all population data we need
    missing_countries = []
    for country in countries:
        if country not in population:
            LOGGER.debug(f"Population data for country '{country}' is missing")
            missing_countries.append(country)

    if not missing_countries:
        LOGGER.info(f"Population data is up-to-date, using population file")
        return population
    else:
        missing_population = scrapePopulation(missing_countries)

    return mergeDicts(population, missing_population)


def scrapePopulation(countries):
    '''
    Extracts realtime country population data from the www
    Returns: dict[country] = population_number
    Outputs: write dict to json file
    '''

    LOGGER.info("Please wait... web scrapping population data")
    pop_url = getIniSetting("url", "population")
    population = {}
    for country in countries:
        name = None
        pop = 0
        match = None
        url = pop_url.format(code=country)
        try:
            handler = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            html = urlopen(handler, timeout=15).read().decode("utf-8")
            match = re.search(
                r"The current population of <strong>(?P<name>[^0-9]+?)</strong> is <strong>(?P<pop>[, 0-9]+?)</strong>", html, re.MULTILINE | re.DOTALL)
            if match is not None:
                name = match.group('name').strip()
                pop = int(match.group('pop').replace(',', '').strip())
                population[country] = pop
                LOGGER.debug(f"{name} current population: {pop}")
        except Exception:
            LOGGER.error(
                f"Getting population data for country '{country}' failed")

    return population


def downloadHistoricalData(type):
    """
    Download total number of covid-19 cases or deaths from the web as csv file
    """
    url = getIniSetting('url', type)
    fn = os.path.join(SCRIPT_PATH, os.path.basename(url))
    if not os.path.isfile(fn):
        LOGGER.debug(f"Downloading new covid-19 {type} file from the web")
        for _ in range(1, 4):
            try:
                handler = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                content = urlopen(handler, timeout=15).read().decode("utf-8")
                with open(fn, 'w') as cvsfile:
                    cvsfile.write(content)
                break
            except Exception:
                continue
        else:
            LOGGER.critical(f"Failed to download csv {type} file")
            sys.exit(DOWNLOAD_FAILED)

    return pd.read_csv(fn)

    #lst_dict = [{}]
    # with open(fn, newline='') as csvfile:
    #    reader = csv.DictReader(csvfile)
    #    for row in reader:
    #        lst_dict.append(row)
    #
    # return lst_dict


# def getNumCases(country, lst_dict):
#     """
#     Get total number of covid-19 cases per country
#     """
#     for row in lst_dict:
#         for key, value in row.items():
#             if key.upper().startswith('country'.upper()):
#                 if value.lower() in country:
#                     numCases[country] = int()


def getPopulation(countries, force):
    """
    Grab new population data if population file 
    does not exists is not up-to-date
    """
    pop_filename = os.path.join(SCRIPT_PATH, "population.json")
    if force:
        population = scrapePopulation(countries)
    else:
        population = checkPopulationFile(countries, pop_filename)
        if not population:
            population = scrapePopulation(countries)

    # write population data to json file
    try:
        with open(pop_filename, 'w') as fp:
            json.dump(population, fp)
    except:
        LOGGER.warning(f"Cannot write to file '{pop_filename}'")
        LOGGER.warning("Please check your file permissions")

    return population


def scrapeCasesDeathsRecoveries(countries):
    """
    Scrape realtime covid-19 data from the web
    """
    cases = {}
    recoveries = {}
    deaths = {}
    url_all = getIniSetting('url', 'all')
    LOGGER.info("Please wait... web scrapping covid-19 data")
    for country in countries:
        url = url_all.format(code=country)
        LOGGER.debug(f"Scraping data for country {country}")
        matches = None
        try:
            handler = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            html = urlopen(handler, timeout=15).read().decode("utf-8")
            matches = re.findall(
                r"<h1>(?P<what>[^0-9]+?):<\/h1>.*?<div.+?<span.*?>(?P<dea>[, 0-9]+?)<\/span>.*?<\/div>", html, re.MULTILINE | re.DOTALL)
            if matches is not None:
                for m in matches:
                    what = m[0].strip()
                    num = int(m[1].replace(',', '').strip())
                    if what.lower() == "Coronavirus Cases".lower():
                        cases[country] = num
                    elif what.lower() == "Deaths".lower():
                        deaths[country] = num
                        LOGGER.debug(
                            "{} current deaths by covid-19: {}".format(country, num))
                    elif what.lower() == "Recovered".lower():
                        recoveries[country] = num
        except Exception:
            LOGGER.error(f"Failed to grab death data for country '{country}'")

    # order asc
    cas_ordered = OrderedDict(sorted(cases.items(), key=lambda kv: kv[1]))
    dea_ordered = OrderedDict(sorted(deaths.items(), key=lambda kv: kv[1]))
    rec_ordered = OrderedDict(sorted(recoveries.items(), key=lambda kv: kv[1]))

    return cas_ordered, dea_ordered, rec_ordered


def hbarPlot(data, title, xlabel, fn):
    """
    Horizontal bar plot function 
    """
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(19, 13))
    vals = list(data.values())
    y_pos = list(range(len(data)))
    ax.barh(y_pos, vals, align='center', color='navy')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data.keys())
    # ax.invert_yaxis()
    nvals = len(vals)
    for i, v in enumerate(vals):
        ax.text(v, i, " ("+str(nvals-i)+") " +
                "{:,}".format(vals[i]), va='center')
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.xaxis.grid(which='major', alpha=0.5)
    ax.xaxis.grid(which='minor', alpha=0.2)
    plt.savefig(fn, bbox_inches='tight')


def shortDateStr(dt):
    """
    Returns this ugly formated ultra short date string
    Please don't blame me, blame guys at CSSEGISandData
    """
    return dt.strftime("X%m/%e/%y").replace('X0', 'X').replace('X', '')


def main():
    # setup logging system first
    setupLogging()

    # setup command line arguments
    args = setupCmdLineArgs()

    # date
    yesterday = datetime.date.today() - datetime.timedelta(1)
    yesterday_str = shortDateStr(yesterday)

    # check external connection
    if not args.internet:
        if not internetConnCheck():
            LOGGER.critical("Internet connection test failed")
            LOGGER.critical(
                "Please check your internet connection and try again later")
            sys.exit(NOT_CONNECTED)

    # get list of countries from text file
    countries = getCountries()

    # scrape (up-to-date) population data per country
    population = getPopulation(countries, force=args.force)

    # download historical number of cases and deaths
    histCases = downloadHistoricalData('cases')
    histDeaths = downloadHistoricalData('deaths')

    # process date command line arg
    dayStr = args.day
    if dayStr is None:
        dayObj = datetime.now()
        dayStr = dayObj.strftime("%Y%m%d")
    else:
        try:
            dayObj = datetime.strptime(dayStr, '%Y%m%d').date()
        except Exception:
            LOGGER.critical(
                "Invalid date format. Please use ISO date without dashes: YYYYMMDD")
            sys.exit(DATE_FMT_ERROR)

    # check if requested date (today, if no date was provided) is already present
    # in the previously downloaded csv file
    if not shortDateStr(dayObj) in histCases.columns:
        LOGGER.debug("Day '{}' not present in downloaded files".format(shortDateStr(dayObj)))
        if dayObj == datetime.now():
            # scrape (today) per country number of cases, deaths, and recoveries
            todayCases, todayDeaths, todayRecoveries = scrapeCasesDeathsRecoveries(
                countries)
        else:
            LOGGER.critical("There is no covid-19 online data available for the selected date")
            LOGGER.critical("First day avail is 20200122. Please input another day and try again")
            sys.exit(NO_DATA_AVAIL)


    # for country in countries:
    #    numCases[country] = getNumCases(country)

    # plots
    baseName = "%s-covid-19-{}.{}" % dayStr
    hbarPlot(todayCases, 'covid-19: number of reported cases per country ({})'.format(dayStr),
             'total number of confirmed cases', baseName.format('cases', 'png'))
    hbarPlot(todayDeaths, 'covid-19: number of reported deaths per country ({})'.format(dayStr),
             'total number of confirmed deaths', baseName.format('deaths', 'png'))
    hbarPlot(todayRecoveries, 'covid-19: number of reported recoveries per country ({})'.format(dayStr),
             'total number of confirmed recoveries', baseName.format('recoveries', 'png'))


if __name__ == '__main__':
    main()
