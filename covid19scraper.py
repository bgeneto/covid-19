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
__date__ = "2020/05/15"
__version__ = "0.0.3"

import os
import re
import sys
import json
import logging
import argparse
import configparser

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path
from multiprocessing import Pool
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from urllib.request import urlopen, Request


# exit error codes
NOT_CONNECTED = 1
FILE_NOT_FOUND = 2
DOWNLOAD_FAILED = 3
DATE_FMT_ERROR = 4
NO_DATA_AVAIL = 5
PERMISSION_ERROR = 6

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
            con = urlopen("http://" + url, timeout=10)
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
    parser.add_argument('--no-con', action='store_true', default=False,
                        dest="no_con", help='Do not check for an active Internet connection')
    parser.add_argument('-f', '--force', action='store_true',
                        default=False, help='force download and regeneration of all data')
    parser.add_argument('--no-parallel', action='store_true', dest='no_parallel',
                        default=False, help='Do not execute some functions in parallel (slower)')
    parser.add_argument('--no-png', action='store_true',
                        dest='no_png', default=False, help='Do not output png image files')
    parser.add_argument('--no-dat', action='store_true', dest='no_dat',
                        default=False, help='Do not output dat files')
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
        with open(os.path.join(SCRIPT_PATH, country_filename)) as file:
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
        LOGGER.info(f"Population data is fine, using existing population file")
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

    LOGGER.info("Please wait... web scrapping population data per country")
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
                LOGGER.info(f"{name} current population: {pop}")
        except Exception:
            LOGGER.error(
                f"Getting population data for country '{country}' failed")

    return population


def downloadHistoricalData(type, dt):
    """
    Download total number of covid-19 cases or deaths from the web as csv file
    and returns the filename
    """
    url = getIniSetting('url', type)
    fn = os.path.join(SCRIPT_PATH, "output", "csv",
                      "{}-{}".format(dt, os.path.basename(url)))
    if not os.path.isfile(fn) or cmdargs.force:
        LOGGER.info(f"Downloading new covid-19 {type} csv file from the web")
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

    return fn


def getPopulation(countries, dt):
    """
    Grab new population data if population file
    does not exists is not up-to-date
    """
    json_filename = os.path.join(
        SCRIPT_PATH, "output", "json", "{}-population.json".format(dt))
    dat_filename = os.path.join(
        SCRIPT_PATH, "output", "dat", "{}-population.dat".format(dt))

    if cmdargs.force:
        population = scrapePopulation(countries)
    else:
        population = checkPopulationFile(countries, json_filename)
        if not population:
            population = scrapePopulation(countries)

    # write population data to file
    try:
        with open(json_filename, 'w') as json_fp:
            json.dump(population, json_fp)
        with open(dat_filename, "w") as dat_fp:
            for key, value in population.items():
                dat_fp.write("%s\t%s\n" % (key, value))
    except Exception as e:
        LOGGER.debug(str(e))
        LOGGER.warning("Cannot write population file")
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


def shortDateStr(dt):
    """
    Returns this ugly formated ultra short date string
    Please don't blame me, blame guys at CSSEGISandData
    """
    return dt.strftime("X%m-%e-%y").replace('X0', 'X').replace('X', '')


def setupOutputFolders():
    LOGGER.debug("Creating output folders")
    try:
        Path(os.path.join(SCRIPT_PATH, "output", "json")).mkdir(
            parents=True, exist_ok=True)
        Path(os.path.join(SCRIPT_PATH, "output", "csv")).mkdir(
            parents=True, exist_ok=True)
        Path(os.path.join(SCRIPT_PATH, "output", "dat")).mkdir(
            parents=True, exist_ok=True)
        Path(os.path.join(SCRIPT_PATH, "output", "png")).mkdir(
            parents=True, exist_ok=True)
    except Exception:
        LOGGER.critical(
            "Unable to create output folders. Please check filesystem permissions")
        sys.exit(PERMISSION_ERROR)


def hbarPlot(df, type):
    """
    Horizontal bar plot function
    """
    title = 'number of reported covid-19 deaths per country'
    xlabel = 'total number of confirmed deaths'
    if type == "cases":
        title = 'number of reported covid-19 cases per country'
        xlabel = 'total number of confirmed cases'
    elif type == "cases-per-mil":
        title = 'number of reported covid-19 cases per million people'
        xlabel = 'total number of confirmed cases per million people'
    elif type == "deaths-per-mil":
        title = 'number of reported covid-19 deaths per million people'
        xlabel = 'total number of confirmed deaths per million people'
    plt.rcdefaults()
    # one plot per day
    for column in df.columns:
        col_name = column.replace('/', '-')
        fn = os.path.join(
            "output", "png", f"{col_name}-{type}-per-country.png")
        if os.path.isfile(fn) and not cmdargs.force:
            continue
        subdf = df[column].sort_values(ascending=True)
        # write to dat file
        dfn = os.path.join(SCRIPT_PATH, "output", "dat",
                           f"{col_name}-{type}-per-country.dat")
        try:
            subdf.to_csv(dfn, sep='\t', encoding='utf-8', header=False)
        except:
            LOGGER.error(
                f"Failed to write {type} .dat file for day '{subdf.name}'")
        vals = list(subdf.values)
        y_pos = list(range(len(subdf)))
        fig, ax = plt.subplots(figsize=(19, 13))
        ax.barh(y_pos, vals, align='center', color='navy')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(subdf.keys(), fontsize=14)
        nvals = len(vals)
        for i, v in enumerate(vals):
            ax.text(v, i, " (P" + str(nvals - i) + ") " +
                    "{:,.2f}".format(float(vals[i])), va='center')
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_title(title, fontsize=18)
        ax.xaxis.grid(which='major', alpha=0.5)
        ax.xaxis.grid(which='minor', alpha=0.2)
        plt.savefig(fn, bbox_inches='tight')
        plt.close()


def genDatFile(type, df, countries):
    """
    Generate historical .dat files for cases and deaths per country
    """
    for country in [c.replace('uk', 'united kingdom') for c in countries]:
        if country in df.index:
            ndf = df.loc[country, :]
            fn = os.path.join(SCRIPT_PATH, "output", "dat",
                              f"{ndf.name}-{type}-historical.dat")
            try:
                ndf.to_csv(fn, sep='\t', encoding='utf-8', header=False)
            except:
                LOGGER.error(
                    f"Failed to write {type} .dat file for country '{ndf.name}'")
        else:
            LOGGER.warning(f"Country '{country}' not found in csv {type} file")


def linePlot(df, title, type, date):
    fn = os.path.join(SCRIPT_PATH, "output", "png",
                      f"{date}-{df.name}-{type}-historical.png")
    if os.path.isfile(fn) and not cmdargs.force:
        return
    plt.rcdefaults()
    # tics interval in days
    interval = 5
    # remove zeroes and nan
    ndf = df.replace(0, np.nan).dropna()
    # max tics
    max_tics = len(ndf)
    ax = ndf.plot.line(title=title,
                       legend=True,
                       xticks=np.arange(0, max_tics, interval))
    ax.set_xlabel("date (m/d/yy)")
    ax.set_ylabel("total number of {}".format(type))
    ax.xaxis.grid(which='major', linestyle='--', alpha=0.5)
    ax.yaxis.grid(which='major', linestyle='--', alpha=0.5)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.xticks(rotation=45)
    plt.savefig(fn, bbox_inches='tight')
    plt.close()


def historicalPlot(type, df, countries, dt):
    """
    Generate historical .png image files for cases and deaths per country
    """
    # plot for selected countries only
    for country in [c.replace('uk', 'united kingdom') for c in countries]:
        if country in df.index:
            linePlot(df.loc[country, :], f'total number of confirmed covid-19 {type}', type, dt)
        else:
            LOGGER.warning(f"Country '{country}' not found in csv {type} file")


def main():
    # we first confirm that your have an active internet connection
    if not cmdargs.no_con:
        if not internetConnCheck():
            LOGGER.critical("Internet connection test failed")
            LOGGER.critical(
                "Please check your internet connection and try again later")
            sys.exit(NOT_CONNECTED)

    # saving the dates
    today = datetime.today()
    yesterday = (today - timedelta(1)).date()
    today = today.date()
    today_str = today.strftime("%Y-%m-%d")
    yesterday_str = shortDateStr(yesterday)

    # get list of countries from user input text file
    countries = getCountries()

    # scrape (up-to-date) population data per country
    population = getPopulation(countries, dt=today_str)

    # download historical number of cases and deaths
    cases_fn = downloadHistoricalData(
        'cases', dt=today_str)
    deaths_fn = downloadHistoricalData(
        'deaths', dt=today_str)

    # convert to csv to df in order to plot
    cases_df = pd.read_csv(cases_fn)
    deaths_df = pd.read_csv(deaths_fn)

    # remove unwanted columns
    del cases_df['Province/State']
    del cases_df['Lat']
    del cases_df['Long']
    del deaths_df['Province/State']
    del deaths_df['Lat']
    del deaths_df['Long']

    # aggregate all country regions
    cases_df = cases_df.groupby(cases_df['Country/Region']).sum()
    deaths_df = deaths_df.groupby(deaths_df['Country/Region']).sum()

    # change to lower case to match countries txt user input file
    cases_df.index = cases_df.index.str.lower()
    deaths_df.index = deaths_df.index.str.lower()

    # historical plots of cases and deaths for selected countries only
    LOGGER.info(f"Please wait, generating per country historical png figures")
    if not cmdargs.no_png:
        if not cmdargs.no_parallel:
            pool.apply_async(historicalPlot, args=(
                'cases', cases_df, countries, yesterday_str))
            pool.apply_async(historicalPlot, args=(
                'deaths', deaths_df, countries, yesterday_str))
        else:
            historicalPlot('cases', cases_df, countries, yesterday_str)
            historicalPlot('deaths', deaths_df, countries, yesterday_str)

    # generate historical dat files for external plot software
    if not cmdargs.no_dat:
        LOGGER.info(
            f"Please wait, generating .dat files for every selected country")
        if not cmdargs.no_parallel:
            pool.apply_async(genDatFile, args=(
                'cases', cases_df, countries,))
            pool.apply_async(genDatFile, args=(
                'deaths', deaths_df, countries,))
        else:
            genDatFile('cases', cases_df, countries)
            genDatFile('deaths', deaths_df, countries)

    # remove not selected countries, accounting for uk name exception
    del_idx = []
    for idx in cases_df.index:
        if idx not in [c.replace('uk', 'united kingdom') for c in countries]:
            del_idx.append(idx)
    cases_df.drop(del_idx, inplace=True)
    deaths_df.drop(del_idx, inplace=True)

    # calculate per mil rates
    cases_per_mil_df = pd.DataFrame().reindex_like(cases_df)
    deaths_per_mil_df = pd.DataFrame().reindex_like(deaths_df)
    for idx in cases_df.index:
        pidx = idx
        if idx == 'united kingdom':
            pidx = 'uk'
        if pidx in population:
            cases_per_mil_df.loc[idx] = 1e6 * cases_df.loc[idx] / population[pidx]
            deaths_per_mil_df.loc[idx] = 1e6 * \
                deaths_df.loc[idx] / population[pidx]

    if not cmdargs.no_png:
        LOGGER.info("Please wait, generating per country bar graph png files")
        LOGGER.info("This may take a long time")
        if not cmdargs.no_parallel:
            pool.apply_async(hbarPlot, args=(
                cases_df, 'cases',))
            pool.apply_async(hbarPlot, args=(
                deaths_df, 'deaths',))
            pool.apply_async(hbarPlot, args=(
                cases_per_mil_df, 'cases-per-mil',))
            pool.apply_async(hbarPlot, args=(
                deaths_per_mil_df, 'deaths-per-mil',))
            pool.close()
            pool.join()
        else:
            hbarPlot(cases_df, 'cases')
            hbarPlot(deaths_df, 'deaths')
            hbarPlot(cases_per_mil_df, 'cases-per-mil')
            hbarPlot(deaths_per_mil_df, 'deaths-per-mil')


if __name__ == '__main__':
    # setup logging system
    setupLogging()

    # setup command line arguments
    cmdargs = setupCmdLineArgs()

    # setup our process pool to run in parallel
    pool = None
    if not cmdargs.no_parallel:
        pool = Pool(processes=4)

    # create output directories
    setupOutputFolders()

    main()
