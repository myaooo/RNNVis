"""
File IO utils
"""

import os
import json
import io
import csv


base_dir = os.path.join(os.path.realpath(__file__), '../../../')


def get_path(path, file_name=None):
    """
    A helper function that get the real/abs path of a file on disk, with the project dir as the base dir.
    Note: there is no checking on the illegality of the args!
    :param path: a relative path to base_dir, optional file_name to use
    :param file_name: an optional file name under the path
    :return: an abs path of the request file / path
    """
    _p = os.path.abspath(os.path.join(base_dir, path))
    if file_name:
        return os.path.join(_p, file_name)
    return _p


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


def dict2json(dict_, file_path=None):
    with io.StringIO() as s_io:
        json.dump(dict_, s_io)
        if file_path is None:
            return s_io.getvalue()
        write2file(s_io, file_path, 'w')


def lists2csv(list_of_list, file_path, delimiter=','):
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
