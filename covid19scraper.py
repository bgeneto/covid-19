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
__date__ = "2020/05/20"
__version__ = "0.1.2"

import os
import re
import sys
import json
import random
import gettext
import logging
import argparse
import configparser

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation

from pathlib import Path
from multiprocessing import Process
from datetime import datetime, timedelta
from collections.abc import Iterable
from collections import defaultdict, OrderedDict
from urllib.request import urlopen, Request
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


# default translation
_ = gettext.gettext

# exit error codes
NOT_CONNECTED = 1
FILE_NOT_FOUND = 2
DOWNLOAD_FAILED = 3
DATE_FMT_ERROR = 4
NO_DATA_AVAIL = 5
PERMISSION_ERROR = 6
MISSING_REQUIREMENT = 7

# script name and path
SCRIPT_PATH = os.path.dirname(os.path.abspath(sys.argv[0]))
SCRIPT_NAME = os.path.splitext(os.path.basename(sys.argv[0]))[0]

# logger name
LOGGER = logging.getLogger(SCRIPT_NAME)

ctrans = None


def iterable(obj):
    return isinstance(obj, Iterable)


def internetConnCheck():
    '''
    Simple internet connection checking by using urlopen.
    Returns True (1) on success or False (0) otherwise.
    '''

    LOGGER.info(_("Checking your internet connection"))

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

    with open(fn, "w", encoding='utf-8') as configFile:
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
    animate_choices = ["gif", "html", "mp4", "png"]
    parser = argparse.ArgumentParser(
        description='This python script scrapes covid-19 data from the web and outputs hundreds '
                    'of graphs for the selected countries in countries.txt file')
    parser.add_argument('-v', '--version', action='version',
                        version=f'%(prog)s v{__version__}')
    parser.add_argument('-a', '--animate', default=None, choices=animate_choices,
                        help='create (html, mp4, png or gif) animated bar racing charts (requires ffmpeg)')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='force download and regeneration of all data')
    parser.add_argument('-l', '--lang', default='en', action="store",
                        help='output messages in your preferred language (es, pt, de, ...)')
    parser.add_argument('-p', '--parallel', action='store_true', default=False, dest='parallel',
                        help='execute faster by running some functions in parallel')
    parser.add_argument('--no-con', action='store_true', default=False, dest="no_con",
                        help='do not check for an active Internet connection')
    parser.add_argument('--no-dat', action='store_true', default=False, dest='no_dat',
                        help='do not output dat files')
    parser.add_argument('--no-png', action='store_true',
                        dest='no_png', default=False, help='do not output png image files')
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
        with open(os.path.join(SCRIPT_PATH, country_filename), 'r', encoding='utf-8') as file:
            countries = [line.strip().lower() for line in file]
    else:
        LOGGER.critical(_("File '{}' not found").format(country_filename))
        LOGGER.critical(
            _("Please double-check the 'country_filename' ini setting and try again"))
        sys.exit(FILE_NOT_FOUND)

    return list(set(countries))


def checkPopulationFile(countries, pop_filename):
    """
    Check if json population file contains all the countries we need
    Returns: None or dict[country] = population_value
    """
    # check if json population file already exists
    population = []
    if os.path.isfile(pop_filename):
        try:
            with open(pop_filename, 'r', encoding='utf-8') as fp:
                population = json.load(fp)
        except:
            LOGGER.error(_("Error reading from file '{}'").format(
                os.path.basename(pop_filename)))
            return None
    else:
        LOGGER.debug(_("Population json file '{}' does not exists").format(
            os.path.basename(pop_filename)))
        return None

    # check if we have all population data we need
    missing_countries = []
    for country in countries:
        if country not in population:
            LOGGER.debug(_("Population data for country '{}' is missing").format(country))
            missing_countries.append(country)

    if not missing_countries:
        LOGGER.info(_("Population data is ok, using existing population file"))
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

    LOGGER.info(_("Please wait... web scrapping population data per country"))
    pop_url = getIniSetting("url", "population")
    population = {}
    for country in countries:
        name = None
        pop = 0
        match = None
        ccode = country.replace(' ', '-')
        url = pop_url.format(code=ccode)
        try:
            handler = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            html = urlopen(handler, timeout=15).read().decode("utf-8")
            match = re.search(
                r"The current population of <strong>(?P<name>[^0-9]+?)</strong> is <strong>(?P<pop>[, 0-9]+?)</strong>", html, re.MULTILINE | re.DOTALL)
            if match is not None:
                name = match.group('name').strip()
                pop = int(match.group('pop').replace(',', '').strip())
                population[country] = pop
                LOGGER.info(_("{} current population: {}").format(translateCountries(country, ctrans), pop))
        except Exception:
            LOGGER.error(
                _("Getting population data for country '{}' failed").format(country))

    return population


def downloadHistoricalData(type, dt):
    """
    Download total number of covid-19 cases or deaths from the web as csv file
    and returns the filename
    """
    url = getIniSetting('url', type['name'])
    fn = os.path.join(SCRIPT_PATH, "output", "csv",
                      "{}-{}".format(dt, os.path.basename(url)))
    if not os.path.isfile(fn) or cmdargs.force:
        LOGGER.info(_("Downloading new covid-19 {} csv file from the web").format(type['trans']))
        for c in range(1, 4):
            try:
                handler = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                content = urlopen(handler, timeout=15).read().decode("utf-8")
                with open(fn, 'w', encoding='utf-8') as cvsfile:
                    cvsfile.write(content)
                break
            except Exception:
                continue
        else:
            LOGGER.critical(_("Failed to download csv {} file").format(type['trans']))
            sys.exit(DOWNLOAD_FAILED)

    return fn


def getPopulation(countries, dt):
    """
    Grab new population data if population file
    does not exists is not up-to-date
    """
    pop_filename = os.path.join(
        SCRIPT_PATH, "output", "json", "population.json")
    json_filename = os.path.join(
        SCRIPT_PATH, "output", "json", "{}-population.json".format(dt))
    dat_filename = os.path.join(
        SCRIPT_PATH, "output", "dat", "{}-population.dat".format(dt))

    if cmdargs.force:
        population = scrapePopulation(countries)
    else:
        population = checkPopulationFile(countries, pop_filename)
        if not population:
            population = scrapePopulation(countries)

    # write population data to json and dat files
    try:
        with open(pop_filename, 'w', encoding='utf-8') as pop_fp:
            json.dump(population, pop_fp)
        with open(json_filename, 'w', encoding='utf-8') as json_fp:
            json.dump(population, json_fp)
        with open(dat_filename, "w", encoding='utf-8') as dat_fp:
            for key, value in population.items():
                dat_fp.write("%s\t%s\n" % (key, value))
    except Exception as e:
        LOGGER.error(_("Cannot write population file"))
        LOGGER.error(_("Please check your file permissions"))
        sys.exit(PERMISSION_ERROR)

    return population


def scrapeCasesDeathsRecoveries(countries):
    """
    Scrape realtime covid-19 data from the web
    """
    cases = {}
    recoveries = {}
    deaths = {}
    url_all = getIniSetting('url', 'all')
    LOGGER.info(_("Please wait... web scrapping covid-19 data"))
    for country in countries:
        url = url_all.format(code=country)
        LOGGER.debug(_("Scraping data for country {}").format(country))
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
                            _("{} current deaths by covid-19: {}").format(country, num))
                    elif what.lower() == "Recovered".lower():
                        recoveries[country] = num
        except Exception:
            LOGGER.error(_("Failed to grab death data for country '{}'").format(country))

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
    LOGGER.debug(_("Creating output folders"))
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
            _("Unable to create output folders. Please check filesystem permissions"))
        sys.exit(PERMISSION_ERROR)


def swapDate(dt):
    """
    Exchange month with day, day with month
    """
    if isinstance(dt, str):
        try:
            m, d, a = dt.split('/')
            dt = d + '/' + m + '/' + a
        except:
            try:
                m, d, a = dt.split('-')
                dt = d + '-' + m + '-' + a
            except:
                pass

    return dt


def translateCountries(clst, ctrans):
    """
    Translate country names if available
    """
    if ctrans is not None:
        if isinstance(clst, str):
            if clst in ctrans:
                return ctrans[clst]
        else:
            for i, c in enumerate(clst):
                if c in ctrans:
                    clst[i] = ctrans[c]

    return clst


def getFlag(ccode):
    im = None
    fn = os.path.join(SCRIPT_PATH, 'input', 'flags', ccode + '.png')
    if os.path.isfile(fn):
        im = plt.imread(fn, format='png')
    else:
        LOGGER.warning(_("File '{}' not found").format(os.path.basename(fn)))

    return im


def addFlag2Plot(coord, ccode, ax):
    """
    Add a flag image to the plot
    """
    img = getFlag(ccode)

    if img is None:
        return

    im = OffsetImage(img, zoom=0.06)
    im.image.axes = ax

    ab = AnnotationBbox(im, coord, xybox=(14, 0), frameon=False,
                        xycoords='data', boxcoords="offset points", pad=0)

    ax.add_artist(ab)


def getCountryCode(dict, key):
    if key in dict:
        return dict[key]

    return None


def hbarPlot(df, type, ginfo, ctrans, ccodes, force):
    """
    Horizontal bar plot function
    """

    # one plot per day
    for column in df.columns:
        col_name = column.replace('/', '-')

        # skip if file exists
        fn = os.path.join(
            "output", "png", f"{col_name}-{type['name']}-per-country.png")
        if os.path.isfile(fn) and not force:
            continue

        # our ordered subset
        subdf = df[column].sort_values(ascending=True)

        # our custom colors
        # color_cycle = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(subdf)))
        color_grad = [(x / float(len(subdf)), 0.0, 0.0, x / float(len(subdf))) for x in (range(len(subdf)))]
        if type['name'] in ['cases', 'cases-per-mil']:
            color_grad = [(0.0, 0.0, x / float(len(subdf)), x / float(len(subdf))) for x in (range(len(subdf)))]

        # write to dat file
        dfn = os.path.join(SCRIPT_PATH, "output", "dat",
                           f"{col_name}-{type['name']}-per-country.dat")
        try:
            subdf.to_csv(dfn, sep='\t', encoding='utf-8', header=False)
        except:
            LOGGER.error(
                _("Failed to write {} .dat file for day '{}'").format(type['trans'], subdf.name))

        vals = list(subdf.values)
        y_pos = list(range(len(subdf)))
        vsize = int(round(len(subdf) / 3))
        fig, ax = plt.subplots(figsize=(round(vsize * 1.77, 2), vsize))
        ax.margins(0.15, 0.01)
        ax.barh(y_pos, vals, align='center', color=color_grad)
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(translateCountries(subdf.keys().tolist(), ctrans), fontsize=14)
        nvals = len(vals)
        # add text and flags to every bar
        for i, v in enumerate(vals):
            val = ginfo['fmt'][type['name']].format(round(vals[i], 2))
            ax.text(v, i, "       " + val + " (P" + str(nvals - i) + ")",
                    va='center', ha='left', fontsize=12)
            if ccodes:
                ccode = getCountryCode(ccodes, subdf.keys()[i])
                if ccode is not None:
                    addFlag2Plot((v, i), ccode, ax)
        ax.set_xlabel(ginfo['label'][type['name']], fontsize=16)
        ax.set_title(ginfo['title'][type['name']].format(subdf.name).upper(), fontsize=18)
        ax.xaxis.grid(which='major', alpha=0.5)
        ax.xaxis.grid(which='minor', alpha=0.2)
        ax.set_facecolor('#EFEEEC')
        plt.savefig(fn, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()


def animatedPlot(i, df, type, fig, ax, colors, ginfo, ctrans, ccodes):
    """
    Horizontal bar plot function
    """

    # our ordered subset
    subdf = df.iloc[:, i].sort_values(ascending=True)
    vals = list(subdf.values)
    y_pos = list(range(len(subdf)))
    ax.clear()
    ax.barh(y_pos, vals, align='center', color=[colors[x] for x in subdf.index.tolist()])
    ax.margins(0.15, 0.01)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', colors='#777777', labelsize=11)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(translateCountries(subdf.keys().tolist(), ctrans), fontsize=14)
    nvals = len(vals)
    for i, v in enumerate(vals):
        val = ginfo['fmt'][type['name']].format(round(vals[i], 2))
        ax.text(v, i, "       " + val + " (P" + str(nvals - i) + ")",
                va='center', ha='left', fontsize=12)
        if ccodes:
            ccode = getCountryCode(ccodes, subdf.keys()[i])
            if ccode is not None:
                addFlag2Plot((v, i), ccode, ax)
    ax.set_title(ginfo['title'][type['name']].format(subdf.name).upper(), fontsize=18)
    ax.xaxis.grid(which='major', alpha=0.5)
    ax.xaxis.grid(which='minor', alpha=0.2)
    plt.box(False)


def genDatFile(type, df, countries):
    """
    Generate historical .dat files for cases and deaths per country
    """
    for country in countries:
        if country in df.index:
            ndf = df.loc[country, :]
            fn = os.path.join(SCRIPT_PATH, "output", "dat",
                              f"{ndf.name}-{type['name']}-historical.dat")
            try:
                ndf.to_csv(fn, sep='\t', encoding='utf-8', header=False)
            except:
                LOGGER.error(
                    _("Failed to write {} .dat file for country '{}'").format(type['trans'], ndf.name))
        else:
            LOGGER.error(_("Country '{}' not found in csv {} file").format(country, type['trans']))


def linePlot(df, type, date, ginfo, ctrans, force):
    fn = os.path.join(SCRIPT_PATH, "output", "png",
                      f"{date}-{df.name}-{type['name']}-historical.png")
    if os.path.isfile(fn) and not force:
        return

    # tics interval in days
    interval = 5

    # remove zeroes and nan
    ndf = df.replace(0, np.nan).dropna()

    # max tics
    max_tics = len(ndf)

    # graph title
    title = ginfo['ltitle'].format(type['trans']).upper()

    # plot line color
    color = 'r'
    if type['name'] in ['cases', 'cases-per-mil']:
        color = 'b'

    # the plot
    plt.figure(figsize=(16, 9))
    ax = ndf.plot.line(legend=True,
                       color=color,
                       xticks=np.arange(0, max_tics, interval))

    # additional visual config
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(ginfo['xlabel'], fontsize=16)
    ax.set_ylabel(ginfo['ylabel'].format(type['trans']), fontsize=16)
    ax.xaxis.grid(which='major', linestyle='--', alpha=0.5)
    ax.yaxis.grid(which='major', linestyle='--', alpha=0.5)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_facecolor('#EFEEEC')
    handles = plt.Rectangle((0, 0), 1, 1, fill=True, color=color)
    ax.legend((handles,), (translateCountries(ndf.name, ctrans),), loc='upper left',
              frameon=False, shadow=False, fontsize='large')
    plt.xticks(rotation=45)
    plt.savefig(fn, bbox_inches='tight')
    plt.close()


def historicalPlot(type, df, countries, dt, ginfo, ctrans, f):
    """
    Generate historical .png image files for cases and deaths per country
    """
    # plot for selected countries only
    for country in countries:
        if country in df.index:
            linePlot(df.loc[country, :], type, dt, ginfo, ctrans, f)
        else:
            LOGGER.error(_("Country '{}' not found in csv {} file").format(country, type['trans']))


def createAnimatedGraph(df, type, fmt, ginfo, ctrans, ccodes):
    """
    Create animated bar racing charts
    """
    # animation begins at day
    bday = 30
    vsize = int(round(len(df) / 3))
    fig, ax = plt.subplots(figsize=(round(vsize * 1.77, 2), vsize))

    # our custom colors
    color_lst = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(len(df))]
    colors = dict(zip(df.index.tolist(), color_lst))
    animator = animation.FuncAnimation(fig, animatedPlot, frames=range(bday, len(df.columns)),
                                       fargs=(df, type, fig, ax, colors, ginfo, ctrans, ccodes),
                                       repeat=False, interval=750)
    fn = os.path.join(SCRIPT_PATH, "output", f"{type['name']}-animated.{fmt}")
    try:
        if fmt == "html":
            with open(fn, "w", encoding='utf-8') as html:
                print(animator.to_html5_video(), file=html)
        elif fmt == "mp4":
            writer = animation.FFMpegWriter(fps=2)
            animator.save(fn, writer=writer)
        elif fmt == "gif":
            writer = animation.PillowWriter(fps=2)
            animator.save(fn, writer=writer, savefig_kwargs={'facecolor': '#EFEEEC'})
        elif fmt == "png":
            from numpngw import AnimatedPNGWriter
            writer = AnimatedPNGWriter(fps=2)
            animator.save(fn, writer=writer, savefig_kwargs={'facecolor': '#EFEEEC'})
    except ModuleNotFoundError:
        LOGGER.critical(_("numpngw package not available! Please install numpngw and try again"))
        LOGGER.critical(_("Tip") + ": pip3 install numpngw")
        sys.exit(MISSING_REQUIREMENT)
    except IndexError:
        LOGGER.critical(_("Pillow package not available! Please install Pillow and try again"))
        LOGGER.critical(_("Tip") + ": pip3 install Pillow")
        sys.exit(MISSING_REQUIREMENT)
    except (FileNotFoundError, RuntimeError):
        LOGGER.critical(_("ffmpeg software not available! Please install ffmpeg and try again"))
        LOGGER.critical(_("Tip") + ": sudo apt update && sudo apt install ffmpeg -y")
        sys.exit(MISSING_REQUIREMENT)


def fmtDates():
    """
    Returns dates as formatted strings
    """
    today = datetime.today()
    yesterday = (today - timedelta(1)).date()
    today = today.date()
    today_str = shortDateStr(today)
    yesterday_str = shortDateStr(yesterday)
    if cmdargs.lang != 'en':
        today_str = swapDate(today_str)
        yesterday_str = swapDate(yesterday_str)

    return (yesterday_str, today_str)


def fmtDataFrameFromCsv(cases_fn, deaths_fn, countries):
    """
    Reads csv data from file and returns formatted/cleared data frame
    """

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

    # change country names to match country names in txt input file
    cases_df.rename(index={'United Kingdom': 'uk'}, inplace=True)
    cases_df.rename(index={'Korea, South': 'south korea'}, inplace=True)
    deaths_df.rename(index={'United Kingdom': 'uk'}, inplace=True)
    deaths_df.rename(index={'Korea, South': 'south korea'}, inplace=True)
    cases_df.index = cases_df.index.str.lower()
    deaths_df.index = deaths_df.index.str.lower()

    # remove unwanted countries
    del_idx = []
    for idx in cases_df.index:
        if idx not in countries:
            del_idx.append(idx)
    cases_df.drop(del_idx, inplace=True)
    deaths_df.drop(del_idx, inplace=True)

    # swap day and month in dates according to the locale/language
    if cmdargs.lang != 'en':
        cases_df.columns = map(swapDate, cases_df.columns)
        deaths_df.columns = map(swapDate, deaths_df.columns)

    return (cases_df, deaths_df)


def main():
    # we first confirm that your have an active internet connection
    if not cmdargs.no_con:
        if not internetConnCheck():
            LOGGER.critical(_("Internet connection test failed"))
            LOGGER.critical(
                _("Please check your internet connection and try again later"))
            sys.exit(NOT_CONNECTED)

    # return dates as formated strings
    yesterday_str, today_str = fmtDates()

    # get list of countries from user input text file
    countries = getCountries()

    # scrape (up-to-date) population data per country
    population = getPopulation(countries, dt=today_str)

    # types of graphs
    type = {'cases': {'name': 'cases', 'trans': _('cases')},
            'deaths': {'name': 'deaths', 'trans': _('deaths')},
            'cases-per-mil': {'name': 'cases-per-mil', 'trans': _('cases-per-mil')},
            'deaths-per-mil': {'name': 'deaths-per-mil', 'trans': _('deaths-per-mil')}
            }

    # download historical number of cases and deaths
    cases_fn = downloadHistoricalData(type['cases'], dt=today_str)
    deaths_fn = downloadHistoricalData(type['deaths'], dt=today_str)

    # convert to csv to df in order to plot
    cases_df, deaths_df = fmtDataFrameFromCsv(cases_fn, deaths_fn, countries)

    # generate historical dat files for external plot software
    if not cmdargs.no_dat:
        LOGGER.info(_("Generating .dat files for every selected country"))
        if cmdargs.parallel:
            p1 = Process(target=genDatFile, args=(
                type['cases'], cases_df, countries))
            p1.start()
            p2 = Process(target=genDatFile, args=(
                type['deaths'], deaths_df, countries))
            p2.start()
            p1.join()
            p2.join()
        else:
            genDatFile(type['cases'], cases_df, countries)
            genDatFile(type['deaths'], deaths_df, countries)

    # graph info: titles, labels, etc...
    ginfo = {
        'title': {
            'deaths': _('number of reported covid-19 deaths per country ({})'),
            'cases': _('number of reported covid-19 cases per country ({})'),
            'cases-per-mil': _('number of reported covid-19 cases per million people ({})'),
            'deaths-per-mil': _('number of reported covid-19 deaths per million people ({})')
        },
        'label': {
            'deaths': _('total number of confirmed deaths'),
            'cases': _('total number of confirmed cases'),
            'cases-per-mil': _('total number of confirmed cases per million people'),
            'deaths-per-mil': _('total number of confirmed deaths per million people')
        },
        'fmt': {
            'deaths': '{:,}',
            'cases': '{:,}',
            'cases-per-mil': '{:,.2f}',
            'deaths-per-mil': '{:,.2f}'
        },
        'ltitle': _('number of confirmed covid-19 {}'),
        'xlabel': _('date (m/d/yy)'),
        'ylabel': _('total number of {}')
    }

    # historical plots of cases and deaths for selected countries only
    LOGGER.info(_("Generating per country historical png figures"))
    if not cmdargs.no_png:
        if cmdargs.parallel:
            p1 = Process(target=historicalPlot,
                         args=(type['cases'], cases_df, countries, yesterday_str,
                               ginfo, ctrans, cmdargs.force))
            p1.start()
            p2 = Process(target=historicalPlot,
                         args=(type['deaths'], deaths_df, countries, yesterday_str,
                               ginfo, ctrans, cmdargs.force))
            p2.start()
            p1.join()
            p2.join()
        else:
            historicalPlot(type['cases'], cases_df, countries, yesterday_str,
                           ginfo, ctrans, cmdargs.force)
            historicalPlot(type['deaths'], deaths_df, countries, yesterday_str,
                           ginfo, ctrans, cmdargs.force)

    # calculate per mil rates
    cases_per_mil_df = pd.DataFrame().reindex_like(cases_df)
    deaths_per_mil_df = pd.DataFrame().reindex_like(deaths_df)
    for idx in cases_df.index:
        if idx in population:
            cases_per_mil_df.loc[idx] = 1e6 * cases_df.loc[idx] / population[idx]
            deaths_per_mil_df.loc[idx] = 1e6 * \
                deaths_df.loc[idx] / population[idx]

    if not cmdargs.no_png:
        LOGGER.info(_("Please wait, generating per country bar graph png files"))
        LOGGER.info(_("This may take a couple of minutes to complete"))
        if cmdargs.parallel:
            p1 = Process(target=hbarPlot, args=(
                cases_df, type['cases'], ginfo, ctrans, ccodes, cmdargs.force))
            p1.start()
            p2 = Process(target=hbarPlot, args=(
                deaths_df, type['deaths'], ginfo, ctrans, ccodes, cmdargs.force))
            p2.start()
            p3 = Process(target=hbarPlot, args=(
                cases_per_mil_df, type['cases-per-mil'], ginfo, ctrans, ccodes, cmdargs.force))
            p3.start()
            p4 = Process(target=hbarPlot, args=(
                deaths_per_mil_df, type['deaths-per-mil'], ginfo, ctrans, ccodes, cmdargs.force))
            p4.start()
            p1.join()
            p2.join()
            p3.join()
            p4.join()
        else:
            LOGGER.info(_("Consider running this stage in parallel (-p option)"))
            hbarPlot(cases_df, type['cases'], ginfo, ctrans, ccodes, cmdargs.force)
            hbarPlot(deaths_df, type['deaths'], ginfo, ctrans, ccodes, cmdargs.force)
            hbarPlot(cases_per_mil_df, type['cases-per-mil'], ginfo, ctrans, ccodes, cmdargs.force)
            hbarPlot(deaths_per_mil_df, type['deaths-per-mil'], ginfo, ctrans, ccodes, cmdargs.force)

    # create animated bar graph racing chart
    if cmdargs.animate:
        LOGGER.info(_("Please wait, creating bar chart race animations"))
        LOGGER.info(_("This may take a couple of minutes to complete"))
        if cmdargs.parallel:
            p1 = Process(target=createAnimatedGraph, args=(
                cases_df, type['cases'], cmdargs.animate, ginfo, ctrans, ccodes))
            p1.start()
            p2 = Process(target=createAnimatedGraph, args=(
                deaths_df, type['deaths'], cmdargs.animate, ginfo, ctrans, ccodes))
            p2.start()
            p3 = Process(target=createAnimatedGraph, args=(
                cases_per_mil_df, type['cases-per-mil'], cmdargs.animate, ginfo, ctrans, ccodes))
            p3.start()
            p4 = Process(target=createAnimatedGraph, args=(
                deaths_per_mil_df, type['deaths-per-mil'], cmdargs.animate, ginfo, ctrans, ccodes))
            p4.start()
            p1.join()
            p2.join()
            p3.join()
            p4.join()
        else:
            LOGGER.info(_("Consider using -p option next time"))
            createAnimatedGraph(cases_df, type['cases'], cmdargs.animate, ginfo, ctrans, ccodes)
            createAnimatedGraph(deaths_df, type['deaths'], cmdargs.animate, ginfo, ctrans, ccodes)
            createAnimatedGraph(cases_per_mil_df, type['cases-per-mil'], cmdargs.animate, ginfo, ctrans, ccodes)
            createAnimatedGraph(deaths_per_mil_df, type['deaths-per-mil'], cmdargs.animate, ginfo, ctrans, ccodes)


if __name__ == '__main__':
    # setup logging system
    setupLogging()

    # setup command line arguments
    cmdargs = setupCmdLineArgs()

    # read country codes in order to show flags
    ccodes = {}
    jfn = os.path.join(SCRIPT_PATH, 'input', 'country-codes-lower-case.json')
    if os.path.isfile(jfn):
        try:
            with open(jfn, 'r', encoding='utf-8') as fp:
                ccodes = json.load(fp)
        except:
            LOGGER.error(_("Error reading from file '{}'").format(
                os.path.basename(jfn)))

    # setup output language
    if cmdargs.lang:
        try:
            lang = gettext.translation(SCRIPT_NAME, localedir=os.path.join(SCRIPT_PATH, 'locale'), languages=[cmdargs.lang])
            lang.install()
            _ = lang.gettext
            # load countries translation file if available
            jfn = os.path.join(SCRIPT_PATH, 'locale', cmdargs.lang, 'countries.json')
            if os.path.isfile(jfn):
                try:
                    with open(jfn, 'r', encoding='utf-8') as fp:
                        ctrans = json.load(fp)
                except Exception as e:
                    LOGGER.error(_("Error reading from file '{}'").format(
                        os.path.basename(jfn)))
            else:
                LOGGER.warning(_("Translation file '{}' not found").format(jfn))
                LOGGER.warning(_("Using default language for country names"))
        except Exception as e:
            LOGGER.warning(f"Unable to find the translation file for the selected ({cmdargs.lang}) language")
            LOGGER.warning("Using the default language (en) instead")

    # create output directories
    setupOutputFolders()

    main()
