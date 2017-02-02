"""
File IO utils
"""

import os
import json
import io
import csv


def write2file(s_io, file_path, mode):
    """
    This is a wrapper function for writing files to disks,
    it will automatically check for dir existence and create dir or file if needed
    :param s_io: a io.StringIO instance or a str
    :param file_path: the path of the file to write to
    :param mode: the writing mode to use
    :return: None
    """
    full_path = os.path.abspath(file_path)
    dir_name = os.path.dirname(full_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(file_path, mode) as f:
        if isinstance(s_io, io.StringIO):
            f.write(s_io.getvalue())
        else:
            f.write(s_io)


def save2json(dict_, file_path):
    with io.StringIO() as s_io:
        json.dump(dict_, s_io)
        write2file(s_io, file_path, 'w')


def save2csv(list_of_list, file_path, delimiter=','):
    with io.StringIO() as s_io:
        writer = csv.writer(s_io, delimiter=delimiter)
        for ls in list_of_list:
            writer.writerow([str(i) for i in ls])
        write2file(s_io, file_path, 'w')


def csv2list(file_path, delimiter=',', mode='r'):
    lists = []
    assert_file_exists(file_path)
    with open(file_path, mode, newline='') as f:
        csv_reader = csv.reader(f, delimiter=delimiter)
        for row in csv_reader:
            lists.append(row)
    return lists


def text2list(file_path, delimiter='|', mode='r'):
    assert_file_exists(file_path)
    with open(file_path, mode) as f:
        s = f.read()
        return s.split(delimiter)


def save2text(a_list, file_path, delimiter='|'):
    s = delimiter.join([str(e) for e in a_list])
    write2file(s, file_path, 'w')


def path_exists(file_or_dir):
    return os.path.exists(file_or_dir)


def file_exists(file_path):
    return os.path.isfile(file_path)


def assert_file_exists(file_path):
    if file_exists(file_path):
        return
    else:
        raise LookupError("Cannot find file: {:s}".format(os.path.abspath(file_path)))


def assert_path_exists(file_or_dir):
    if os.path.exists(file_or_dir):
        return
    else:
        raise LookupError("Cannot find file or dir: {:s}".format(os.path.abspath(file_or_dir)))
