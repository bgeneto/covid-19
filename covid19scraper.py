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
__date__ = "2020/06/16"
__version__ = "0.2.2"

import argparse
import configparser
import gettext
import json
import logging
import os
import random
import re
import sys

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from multiprocessing import Process
from pathlib import Path
from urllib.request import Request, urlopen
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


# default translation
_ = gettext.gettext

# exit error codes
NOT_CONNECTED = 1
FILE_NOT_FOUND = 2
DOWNLOAD_FAILED = 3
FILE_READ_ERROR = 4
NO_DATA_AVAIL = 5
PERMISSION_ERROR = 6
MISSING_REQUIREMENT = 7
COUNTRY_NOT_FOUND = 8

# script name and path
SCRIPT_PATH = os.path.dirname(os.path.abspath(sys.argv[0]))
SCRIPT_NAME = os.path.splitext(os.path.basename(sys.argv[0]))[0]

# logger name
LOGGER = logging.getLogger(SCRIPT_NAME)


def connectionCheck():
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
            return
        except Exception:
            continue

    # test failed, terminate script execution
    LOGGER.critical(_("Internet connection test failed"))
    LOGGER.critical(
        _("Please check your internet connection and try again later"))
    sys.exit(NOT_CONNECTED)


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
    animate_choices = ["gif", "html", "mp4", "png", "none"]
    graph_choices = ["all", "latest", "none"]
    parser = argparse.ArgumentParser(
        description='This python script scrapes covid-19 data from the web and outputs hundreds '
                    'of graphs for the selected countries in countries.txt file')
    parser.add_argument('-v', '--version', action='version',
                        version=f'%(prog)s v{__version__}')
    parser.add_argument('-a', '--anim', default='html', choices=animate_choices,
                        help='create (html, mp4, png or gif) animated bar racing charts (requires ffmpeg)')
    parser.add_argument('-d', '--dat', action='store_true', default=False,
                        help='output dat files')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='force download and regeneration of all data and graphs')
    parser.add_argument('-g', '--graph', default='latest', choices=graph_choices,
                        help='output line and bar graph files (all = every day)')
    parser.add_argument('-l', '--lang', default='en', action="store",
                        help='output messages in your preferred language (es, de, pt, ...)')
    parser.add_argument('--no-con', action='store_true', default=False, dest="no_con",
                        help='do not check for an active Internet connection')
    parser.add_argument('-p', '--parallel', action='store_true', default=False, dest='parallel',
                        help='parallel execution (min. 6 cores, 8GB RAM)')
    parser.add_argument('-s', '--smooth', action='store_true', default=False,
                        help='smooth animation transitions by interpolating data')
    args = parser.parse_args()

    return args


def getCountryNames():
    '''
    Read country names from text file
    Returns: list of names
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


def getCountryCodes(cnames):
    """
    Read country codes from json file
    Returns: pandas dataframe with codes as indexes and country name as column
    """
    resdict = {}
    jfn = os.path.join(SCRIPT_PATH, 'locale', 'en',
                       'countries-translation.json')
    cdict = loadJsonFile(jfn)
    if not cdict:
        LOGGER.critical(_("Country translation file not found or corrupted"))
        LOGGER.critical("({})".format(jfn))
        sys.exit(FILE_NOT_FOUND)

    # now find the codes
    for name in cnames:
        for key, value in cdict.items():
            if name == value.strip().lower():
                resdict[key] = {'name': value}
                break
        else:
            LOGGER.error(_("Not a valid country name ({})").format(name))
            LOGGER.error(_("Check your countries input file"))
            sys.exit(COUNTRY_NOT_FOUND)

    # last resort validation
    if len(cnames) != len(resdict):
        LOGGER.error(_("Failed getting country codes"))
        sys.exit(COUNTRY_NOT_FOUND)

    return pd.DataFrame.from_dict(resdict, orient='index', columns=['name'])


def checkPopulationFile(country_df, jfn):
    """
    Check if json population file contains all the countries we need
    Returns: None or dict[country] = population_value
    """
    # check if json population file already exists
    population = loadJsonFile(jfn)
    if not population:
        return False

    # check if we have all population data we need
    missing = False
    for ccode in country_df.index:
        if ccode not in population:
            LOGGER.warning(_("Population data for country '{}' is missing").format(
                country_df.loc[ccode, 'translation']))
            missing = True

    if missing:
        return False

    LOGGER.info(_("Using existing population data file"))

    return population


def scrapePopulation(country_df):
    '''
    Extracts realtime country population data from the www
    Returns: dict[country] = population_number
    Outputs: write dict to json file
    '''

    LOGGER.info(_("Please wait... web scrapping population data per country"))
    pop_url = getIniSetting("url", "population")
    population = {}
    for country in [c.replace('United States', 'us').replace('United Kingdom', 'uk') for c in country_df['name']]:
        cfullname, match = None, None
        pop = 0
        ccode = country.replace(' ', '-').lower()
        url = pop_url.format(code=ccode)
        try:
            handler = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            html = urlopen(handler, timeout=15).read().decode("utf-8")
            match = re.search(
                r"The current population of <strong>(?P<name>[^0-9]+?)</strong> is <strong>(?P<pop>[, 0-9]+?)</strong>", html, re.MULTILINE | re.DOTALL)
            if match is not None:
                cfullname = match.group('name').strip()
                name = re.sub(r'\bus\b', 'United States', country)
                name = re.sub(r'\buk\b', 'United Kingdom', name)
                ccode = country_df.index[country_df['name'] == name][0]
                pop = int(match.group('pop').replace(',', '').strip())
                population[ccode] = pop
                LOGGER.info(_("{} current population: {}").format(
                    country_df.loc[ccode, 'translation'], pop))
        except Exception as e:
            LOGGER.error(str(e))
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
                      "{}_{}".format(dt, os.path.basename(url)))
    if not os.path.isfile(fn) or cmdargs.force:
        LOGGER.info(
            _("Downloading new covid-19 {} csv file from the web").format(type['trans']))
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
            LOGGER.critical(
                _("Failed to download csv {} file").format(type['trans']))
            sys.exit(DOWNLOAD_FAILED)

    return fn


def getCountryPopulation(country_df, dt):
    """
    Grab new population data if population file
    does not exists or is not up-to-date (missing country)
    """
    pop_filename = os.path.join(
        SCRIPT_PATH, "output", "json", "population.json")
    json_filename = os.path.join(
        SCRIPT_PATH, "output", "json", "{}_population.json".format(dt))
    dat_filename = os.path.join(
        SCRIPT_PATH, "output", "dat", "{}_population.dat".format(dt))

    if cmdargs.force:
        population = scrapePopulation(country_df)
    else:
        population = checkPopulationFile(country_df, pop_filename)
        if not population:
            population = scrapePopulation(country_df)

    # write population data to json and dat files
    try:
        with open(pop_filename, 'w', encoding='utf-8') as pop_fp:
            json.dump(population, pop_fp)
        with open(json_filename, 'w', encoding='utf-8') as json_fp:
            json.dump(population, json_fp)
        if cmdargs.dat:
            with open(dat_filename, "w", encoding='utf-8') as dat_fp:
                for key, value in population.items():
                    dat_fp.write("%s\t%s\n" % (key, value))
    except:
        LOGGER.error(_("Cannot write population file"))
        LOGGER.error(_("Please check your file permissions"))
        sys.exit(PERMISSION_ERROR)

    country_df["population"] = pd.Series(population)


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


def getFlag(ccode):
    im = None
    fn = os.path.join(SCRIPT_PATH, 'input', 'flags', ccode + '.png')
    if os.path.isfile(fn):
        im = plt.imread(fn, format='png')
    else:
        LOGGER.warning(_("File '{}' not found").format(os.path.basename(fn)))

    return im


def addFlag2Plot(coord, code, ax, zoom=0.065, xbox=14):
    """
    Add a flag image to the plot
    """
    img = getFlag(code)

    if img is None:
        return

    im = OffsetImage(img, zoom=zoom)
    im.image.axes = ax

    ab = AnnotationBbox(im, coord, xybox=(xbox, 0), frameon=False,
                        xycoords='data', boxcoords="offset points", pad=0)

    ax.add_artist(ab)


def setupHbarPlot(df, cdf, type, ginfo, ax, color):
    vals = list(df.values)
    y_pos = list(range(len(df)))

    ax.text(0.985, 0.06, '© 2020 bgeneto', transform=ax.transAxes, ha='right',
            color='#777777', bbox=dict(facecolor='white', alpha=0.75, edgecolor='white'))
    ax.text(0.985, 0.02, 'Fonte: www.worldometers.info', transform=ax.transAxes, ha='right',
            color='#777777', bbox=dict(facecolor='white', alpha=0.75, edgecolor='white'))

    ax.margins(0.15, 0.01)
    ax.barh(y_pos, vals, align='center', color=color)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_yticks(y_pos)
    # uggly but required in order to keep original keys order
    uggly = pd.DataFrame({'name': df.keys()}).merge(
        cdf[['name', 'translation']])['translation'].values
    ax.set_yticklabels(uggly, fontsize=14)
    nvals = len(vals)
    # add text and flags to every bar
    for i, v in enumerate(vals):
        val = ginfo['fmt'][type['name']].format(round(vals[i], 2))
        ax.text(v, i, "       " + val + " (P" + str(nvals - i) + ")",
                va='center', ha='left', fontsize=12)
        ccode = cdf[cdf['name'] == df.keys()[i]].index[0]
        addFlag2Plot((v, i), ccode, ax)
    ax.set_xlabel(ginfo['label'][type['name']], fontsize=16)
    ax.set_title(ginfo['title'][type['name']].format(
                 df.name).upper(), fontsize=18)
    ax.xaxis.grid(which='major', alpha=0.5)
    ax.xaxis.grid(which='minor', alpha=0.2)


def hbarPlot(df, type, ginfo, cdf, cmdargs):
    """
    Horizontal bar plot function
    """
    # last day plot only
    columns = [df.columns[-1]]
    if 'all' in cmdargs.graph:
        # one plot per day
        columns = df.columns

    for column in columns:
        col_name = column.replace('/', '-')

        # skip if file already exists
        fn = os.path.join(
            "output", "png", f"{col_name}_{type['name']}_per_country.png")
        if os.path.isfile(fn) and not cmdargs.force:
            continue

        # hbar: from highest to lowest
        subdf = df[column].sort_values(ascending=True)

        # our custom colors
        rows = len(subdf)
        # color_cycle = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, rows))
        color_grad = []
        cmap = plt.get_cmap('coolwarm')
        for x in (range(rows)):
            color_grad.append(cmap(1. * x / rows))
            #frac = x / float(rows)
            #frac = 0.15 if frac < 0.15 else frac
            # if 'cases' in type['name']:
            #    color_grad.append((0.0, 0.0, frac, frac))
            #
            # else:
            #    color_grad.append((frac, 0.0, 0.0, frac))

        # write to dat file
        if cmdargs.dat:
            dfn = os.path.join(SCRIPT_PATH, "output", "dat",
                               f"{col_name}_{type['name']}_per_country.dat")
            try:
                subdf.to_csv(dfn, sep='\t', encoding='utf-8', header=False)
            except:
                LOGGER.error(
                    _("Failed to write {} .dat file for day '{}'").format(type['trans'], subdf.name))

        # auto size w x h
        vsize = int(round(rows / 3))
        vsize = vsize if vsize > 9 else 9
        fig, ax = plt.subplots(figsize=(round(vsize * 1.77, 2), vsize))
        setupHbarPlot(subdf, cdf, type, ginfo, ax, color_grad)
        ax.set_facecolor('#EFEEEC')
        plt.savefig(fn, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close('all')


def setupHbarPlot2(vals, y_pos, ylabels, cdf, type, ginfo, ax, color, dtfmt):
    ax.margins(0.15, 0.01)
    ax.barh(y_pos, vals, align='center', color=color)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ylabels, fontsize=14)
    nvals = len(vals)
    # credits
    ax.text(0.985, 0.06, '© 2020 bgeneto', transform=ax.transAxes, ha='right',
            color='#777777', bbox=dict(facecolor='white', alpha=0.75, edgecolor='white'))
    ax.text(0.985, 0.02, 'Fonte: www.worldometers.info', transform=ax.transAxes, ha='right',
            color='#777777', bbox=dict(facecolor='white', alpha=0.75, edgecolor='white'))

    fmt = ginfo['fmt'][type]
    # add text and flags to every bar
    space = "       "
    zoom = 0.06
    xbox = 14
    if nvals < 12:
        space = "             "
        zoom = 0.12
        xbox = 26
    for name, x, y in zip(vals.index, vals.values, y_pos.values):
        code = cdf[cdf['name']==name].index[0].lower()
        val = fmt.format(round(x, 2))
        pos = int(round(nvals - y + 1))
        ax.text(x, y, space + val + " (P" + str(pos) + ")",
                va='center', ha='left', fontsize=12)
        addFlag2Plot((x, y), code, ax, zoom, xbox)

    ax.set_xlabel(ginfo['label'][type['name']], fontsize=16)
    ax.set_title(ginfo['title'][type['name']].format(
                 vals.name.strftime(dtfmt)).upper(), fontsize=18)
    ax.xaxis.grid(which='major', alpha=0.5)
    ax.xaxis.grid(which='minor', alpha=0.2)


def animatedPlot(i, df, df_rank, type, fig, ax, colors, ginfo, cdf, dtfmt):
    """
    Horizontal bar plot function
    """

    # our ordered subset
    ax.clear()
    ax.xaxis.set_ticks_position('top')
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', colors='#777777', labelsize=11)
    # avoid cutting y-labels (state name)
    plt.gcf().subplots_adjust(left=0.20)
    plt.box(False)
    color = [colors[x] for x in cdf.index.tolist()]
    vals = df.iloc[i]
    y_pos = df_rank.iloc[i]
    ylabels = [cdf[cdf['name'] == c].translation.values[0] for c in df.columns.values]
    setupHbarPlot2(vals, y_pos, ylabels, cdf, type, ginfo, ax, color, dtfmt)
    ax.set_xlabel(None)


def genDatFile(type, df, cdf):
    """
    Generate historical .dat files for cases and deaths per country
    """
    for country in cdf['name']:
        if country in df.index:
            ndf = df.loc[country, :]
            fn = os.path.join(SCRIPT_PATH, "output", "dat",
                              f"{ndf.name}_{type['name']}_historical.dat")
            try:
                ndf.to_csv(fn, sep='\t', encoding='utf-8', header=False)
            except:
                LOGGER.error(
                    _("Failed to write {} .dat file for country '{}'").format(type['trans'], ndf.name))
        else:
            LOGGER.error(_("Country '{}' not found in csv {} file").format(
                country, type['trans']))


def linePlot(df, type, ginfo, cdf, cmdargs):
    # tics interval in days
    interval = 5

    # remove zeroes and nan
    ndf = df.replace(0, np.nan).dropna()

    # skip if file already exists
    date = ndf.index[-1].replace('/', '-')
    fn = os.path.join(SCRIPT_PATH, "output", "png",
                      f"{date}_{df.name}_{type['name']}_historical.png")
    if os.path.isfile(fn) and not cmdargs.force:
        return

    # max tics
    max_tics = len(ndf)

    # graph title
    title = ginfo['ltitle'].format(type['trans']).upper()

    # plot line color
    color = 'r'
    if 'cases' in type['name']:
        color = 'b'

    # the plot
    hsize = max_tics / 9
    vsize = hsize / 2
    plt.figure(figsize=(hsize, vsize))
    ax = ndf.plot.line(legend=True,
                       color=color,
                       xticks=np.arange(0, max_tics, interval))

    # additional visual config
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(ginfo['xlabel'], fontsize='large')
    ax.set_ylabel(ginfo['ylabel'].format(type['trans']), fontsize='large')
    ax.xaxis.grid(which='major', linestyle='--', alpha=0.5)
    ax.yaxis.grid(which='major', linestyle='--', alpha=0.5)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_facecolor('#EFEEEC')
    handles = plt.Rectangle((0, 0), 1, 1, fill=True, color=color)
    ax.legend((handles,), (cdf.loc[cdf['name'] == ndf.name, 'translation']), loc='upper left',
              frameon=False, shadow=False, fontsize='large')
    plt.xticks(rotation=45)
    plt.savefig(fn, bbox_inches='tight')
    plt.close('all')


def historicalPlot(type, df, ginfo, countries_df, cmdargs):
    """
    Generate historical .png image files for cases and deaths per country
    """
    # plot for selected countries only
    for country in countries_df['name'].values:
        if country in df.index:
            linePlot(df.loc[country, :], type,
                     ginfo, countries_df, cmdargs)
        else:
            LOGGER.error(_("Country '{}' not found in csv {} file").format(
                country, type['trans']))


def createAnimatedGraph(df, type, ginfo, countries_df, cmdargs):
    """
    Create animated bar racing charts
    """
    # animation begins at day
    bday = 39
    vsize = int(round(len(df) / 3))
    fig, ax = plt.subplots(figsize=(round(vsize * 1.77, 2), vsize))

    # create a new df to work with (dates in rows, countries in columns)
    ndf = df.T.interpolate()

    # convert index to datetime
    dtfmt = '%d/%m/%y' if cmdargs.lang == 'pt' else '%m/%d/%y'
    ndf.index = pd.to_datetime(ndf.index, format=dtfmt)

    # add interpolated data to smooth transitions
    steps = 5 if cmdargs.smooth else 1
    bday = bday*steps
    num_periods = (len(ndf) - 1) * steps + 1
    dr = pd.date_range(start=ndf.index[0],
                       end=ndf.index[-1], periods=num_periods)
    ndf = ndf.reindex(dr)
    ndf.index.name = 'date'
    ndf_rank_expanded = ndf.rank(axis=1)
    ndf = ndf.interpolate()
    ndf_rank_expanded = ndf_rank_expanded.interpolate()

    # our custom colors
    color_lst = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(len(df))]
    colors = dict(zip(countries_df.index, color_lst))
    animator = animation.FuncAnimation(fig, animatedPlot, frames=range(bday, len(ndf)),
                                       fargs=(ndf, ndf_rank_expanded, type, fig, ax,
                                              colors, ginfo, countries_df, dtfmt),
                                       repeat=False, interval=int(round(1000 / steps)))

    fn = os.path.join(SCRIPT_PATH, "output",
                      f"{type['name']}_animated_{cmdargs.lang}.{cmdargs.anim}")

    try:
        if cmdargs.anim == "html":
            with open(fn, "w", encoding='utf-8') as html:
                print(animator.to_html5_video(), file=html)
        elif cmdargs.anim == "mp4":
            writer = animation.FFMpegWriter(fps=steps)
            animator.save(fn, writer=writer)
        elif cmdargs.anim == "gif":
            writer = animation.PillowWriter(fps=steps)
            animator.save(fn, writer=writer, savefig_kwargs={
                          'facecolor': '#EFEEEC'})
        elif cmdargs.anim == "png":
            from numpngw import AnimatedPNGWriter
            writer = AnimatedPNGWriter(fps=steps)
            animator.save(fn, writer=writer, savefig_kwargs={
                          'facecolor': '#EFEEEC'})
    except ModuleNotFoundError:
        LOGGER.critical(
            _("numpngw package not available! Please install numpngw and try again"))
        LOGGER.critical(_("Tip") + ": pip3 install numpngw")
        sys.exit(MISSING_REQUIREMENT)
    except IndexError:
        LOGGER.critical(
            _("Pillow package not available! Please install Pillow and try again"))
        LOGGER.critical(_("Tip") + ": pip3 install Pillow")
        sys.exit(MISSING_REQUIREMENT)
    except (FileNotFoundError, RuntimeError):
        LOGGER.critical(
            _("ffmpeg software not available! Please install ffmpeg and try again"))
        LOGGER.critical(
            _("Tip") + ": sudo apt update && sudo apt install ffmpeg -y")
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


def fmtDataFrameFromCsv(cases_fn, deaths_fn, cdf):
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
    cases_df.rename(index={'US': 'United States'}, inplace=True)
    deaths_df.rename(index={'US': 'United States'}, inplace=True)
    cases_df.rename(index={'Korea, South': 'South Korea'}, inplace=True)
    deaths_df.rename(index={'Korea, South': 'South Korea'}, inplace=True)
    #cases_df.index = cases_df.index.str.lower()
    #deaths_df.index = deaths_df.index.str.lower()

    # remove unwanted countries
    del_idx = []
    for idx in cases_df.index:
        if idx not in cdf['name'].values:
            del_idx.append(idx)
    cases_df.drop(del_idx, inplace=True)
    deaths_df.drop(del_idx, inplace=True)

    # swap day and month in dates according to the locale/language
    if cmdargs.lang != 'en':
        cases_df.columns = map(swapDate, cases_df.columns)
        deaths_df.columns = map(swapDate, deaths_df.columns)

    return (cases_df, deaths_df)


def readCountryCodes():
    ccodes = None
    jfn = os.path.join(SCRIPT_PATH, 'input', 'country-codes-lower-case.json')
    ccodes = loadJsonFile(jfn)
    if not ccodes:
        LOGGER.error(_("Error reading from file '{}'").format(
                     os.path.basename(jfn)))

    return ccodes


def getCountryArea(country_df):
    jfn = os.path.join(SCRIPT_PATH, 'input', 'country-area.json')
    careas = loadJsonFile(jfn)
    if not careas:
        LOGGER.error(_("Error reading from file '{}'").format(
                     os.path.basename(jfn)))
        return None

    for index, row in country_df.iterrows():
        if row['name'] in careas:
            country_df.loc[index, 'area'] = careas[row['name']]
        else:
            LOGGER.error(_("Area not found for country '{}'").format(
                row['translation']))


def loadJsonFile(jfn):
    """
    Read a json file and return a corresponding object
    """
    resjson = None
    if os.path.isfile(jfn):
        try:
            with open(jfn, 'r', encoding='utf-8') as fp:
                resjson = json.load(fp)
        except:
            LOGGER.error(_("Error reading from file '{}'").format(
                os.path.basename(jfn)))
    else:
        LOGGER.debug(_("JSON file '{}' not found").format(jfn))

    return resjson


def getCountryTranslation(df):
    # load countries translation file if available
    jfn = os.path.join(SCRIPT_PATH, 'locale', cmdargs.lang,
                       'countries-translation.json')
    ctrans = loadJsonFile(jfn)
    if not ctrans:
        LOGGER.warning(_("Country translation file not found or corrupted"))
        LOGGER.warning(_("Using the default language"))
        ctrans = df['name']

    df["translation"] = pd.Series(ctrans)


def setupTranslation():
    global _
    try:
        lang = gettext.translation(SCRIPT_NAME, localedir=os.path.join(
            SCRIPT_PATH, 'locale'), languages=[cmdargs.lang])
        lang.install()
        _ = lang.gettext
    except:
        LOGGER.warning(
            f"Unable to find the translation file for language '{cmdargs.lang}'")
        LOGGER.warning("Using the default language")


def main():
    # we first confirm that your have an active internet connection
    if not cmdargs.no_con:
        connectionCheck()

    # store dates as formated strings
    yesterday_str, today_str = fmtDates()

    # get list of country names from user input text file
    country_names = getCountryNames()

    # get country code (index) from country name (column)
    countries_df = getCountryCodes(country_names)
    del country_names

    # get translation text for country names
    getCountryTranslation(countries_df)

    # scrape (up-to-date) population data per country
    getCountryPopulation(countries_df, dt=today_str)

    # read country areas in km2 in order to compute population density
    getCountryArea(countries_df)

    # compute population density
    countries_df['density'] = countries_df['population'] / countries_df['area']

    # types of graphs
    gtype = {'cases': {'name': 'cases', 'trans': _('cases')},
             'deaths': {'name': 'deaths', 'trans': _('deaths')},
             'cases_per_mil': {'name': 'cases_per_mil', 'trans': 'cases_per_mil'},
             'deaths_per_mil': {'name': 'deaths_per_mil', 'trans': 'deaths_per_mil'},
             'cases_per_den': {'name': 'cases_per_den', 'trans': 'cases_per_den'},
             'deaths_per_den': {'name': 'deaths_per_den', 'trans': 'deaths_per_den'}
             }

    # download historical number of cases and deaths
    cases_fn = downloadHistoricalData(gtype['cases'], dt=today_str)
    deaths_fn = downloadHistoricalData(gtype['deaths'], dt=today_str)

    # dict of dataframes
    df = {}

    # convert to csv to dataframes in order to plot
    df['cases'], df['deaths'] = fmtDataFrameFromCsv(
        cases_fn, deaths_fn, countries_df)

    # generate historical dat files for external plot software
    if cmdargs.dat:
        LOGGER.info(_("Generating .dat files for every selected country"))
        if cmdargs.parallel:
            p1 = Process(target=genDatFile, args=(
                gtype['cases'], df['cases'], countries_df))
            p1.start()
            p2 = Process(target=genDatFile, args=(
                gtype['deaths'], df['deaths'], countries_df))
            p2.start()
            p1.join()
            p2.join()
        else:
            genDatFile(gtype['cases'], df['cases'], countries_df)
            genDatFile(gtype['deaths'], df['deaths'], countries_df)

    # graph info: titles, labels, etc...
    ginfo = {
        'title': {
            'deaths': _('number of reported covid-19 deaths per country ({})'),
            'cases': _('number of reported covid-19 cases per country ({})'),
            'cases_per_mil': _('number of reported covid-19 cases per million people ({})'),
            'deaths_per_mil': _('number of reported covid-19 deaths per million people ({})'),
            'cases_per_den': _('number of reported covid-19 cases per population density ({})'),
            'deaths_per_den': _('number of reported covid-19 deaths per population density ({})')
        },
        'label': {
            'deaths': _('total number of confirmed deaths'),
            'cases': _('total number of confirmed cases'),
            'cases_per_mil': _('total number of confirmed cases per million people'),
            'deaths_per_mil': _('total number of confirmed deaths per million people'),
            'cases_per_den': _('total number of confirmed cases per population density'),
            'deaths_per_den': _('total number of confirmed deaths per population density')
        },
        'fmt': {
            'deaths': '{:,.0f}',
            'cases': '{:,.0f}',
            'cases_per_mil': '{:,.2f}',
            'deaths_per_mil': '{:,.2f}',
            'cases_per_den': '{:,.2f}',
            'deaths_per_den': '{:,.2f}'
        },
        'ltitle': _('number of confirmed covid-19 {}'),
        'xlabel': _('(m/d/y)'),
        'ylabel': _('total number of {}')
    }

    # historical plots of cases and deaths for selected countries only
    if 'none' not in cmdargs.graph:
        LOGGER.info(_("Generating per country historical png figures"))
        if cmdargs.parallel:
            p1 = Process(target=historicalPlot,
                         args=(gtype['cases'], df['cases'],
                               ginfo, countries_df, cmdargs))
            p1.start()
            p2 = Process(target=historicalPlot,
                         args=(gtype['deaths'], df['deaths'],
                               ginfo, countries_df, cmdargs))
            p2.start()
            p1.join()
            p2.join()
        else:
            historicalPlot(gtype['cases'], df['cases'],
                           ginfo, countries_df, cmdargs)
            historicalPlot(gtype['deaths'], df['deaths'],
                           ginfo, countries_df, cmdargs)

    # calculate per mil rates and per (population) density rates
    df['cases_per_mil'] = pd.DataFrame().reindex_like(df['cases'])
    df['deaths_per_mil'] = pd.DataFrame().reindex_like(df['deaths'])
    df['cases_per_den'] = pd.DataFrame().reindex_like(df['cases'])
    df['deaths_per_den'] = pd.DataFrame().reindex_like(df['deaths'])
    for idx in df['cases'].index:
        df['cases_per_mil'].loc[idx] = 1e6 * \
            df['cases'].loc[idx] / countries_df.loc[countries_df['name']
                                                    == idx, 'population'][0]
        df['deaths_per_mil'].loc[idx] = 1e6 * \
            df['deaths'].loc[idx] / countries_df.loc[countries_df['name']
                                                     == idx, 'population'][0]
        df['cases_per_den'].loc[idx] = df['cases'].loc[idx] / \
            countries_df.loc[countries_df['name'] == idx, 'density'][0]
        df['deaths_per_den'].loc[idx] = df['deaths'].loc[idx] / \
            countries_df.loc[countries_df['name'] == idx, 'density'][0]

    if 'none' not in cmdargs.graph:
        LOGGER.info(
            _("Please wait, generating per country bar graph png files"))
        if cmdargs.parallel:
            proc = []
            for t in gtype.keys():
                p = Process(target=hbarPlot, args=(
                    df[t], gtype[t], ginfo, countries_df, cmdargs))
                p.start()
                proc.append(p)
            # join all process
            for p in proc:
                p.join()
        else:
            if 'all' in cmdargs.graph:
                LOGGER.info(_("This may take a couple of minutes to complete"))
                LOGGER.info(
                    _("Consider running this stage in parallel (-p option)"))
            for t in gtype.keys():
                hbarPlot(df[t], gtype[t], ginfo, countries_df, cmdargs)

    # create animated bar graph racing chart
    if 'none' not in cmdargs.anim:
        LOGGER.info(_("Please wait, creating bar chart race animations"))
        LOGGER.info(_("This may take a couple of minutes to complete"))
        if cmdargs.parallel:
            proc = []
            for t in gtype.keys():
                p = Process(target=createAnimatedGraph, args=(
                    df[t], gtype[t], ginfo, countries_df, cmdargs))
                p.start()
                proc.append(p)
            # join all processes
            for p in proc:
                p.join()
        else:
            LOGGER.info(_("Consider using -p option next time"))
            for t in gtype.keys():
                createAnimatedGraph(
                    df[t], gtype[t], ginfo, countries_df, cmdargs)


if __name__ == '__main__':
    # setup logging system
    setupLogging()

    # setup command line arguments
    cmdargs = setupCmdLineArgs()

    # setup output language
    setupTranslation()

    # create output directories
    setupOutputFolders()

    main()
