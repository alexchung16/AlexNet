#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tools.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/3/26 下午5:20
# @ Software   : PyCharm
#-------------------------------------------------------

import math
import sys
import os
import shutil


def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r{0}:[{1}{2}]{3}%\t{4}/{5}'.format (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()


def makedir(path):
    """
    create dir
    :param path:
    :return:
    """
    try:
        if os.path.exists(path) is False:
            os.makedirs(path)
            print('{0} has been created'.format(path))
    except Exception as e:
        raise e


def refresh_dir(path):

    try:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        print('{0} has been refreshed'.format(path))
    except Exception as e:
        raise e
