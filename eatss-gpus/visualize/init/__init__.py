import os
from configparser import ConfigParser


def init_config():
    cfg = ConfigParser()
    config_file = 'config.ini'
    filename = cfg.read(config_file)
    if filename is []:
        raise Exception("configuration file {0:} not found".format(config_file))

    data_dir = cfg['DEFAULT']['dataFolder']
    figure_dir = cfg['DEFAULT']['figureFolder']
    figure_list_str = cfg['FOLDERNAMES']['folders']
    try:
        figure_list = figure_list_str.lstrip('[').rstrip(']').replace('\n', '').split(',')
    except ValueError:
        raise Exception("incorrect string submitted as graph name")

    def create_directory_structure(root, filenames):
        try:
            os.makedirs(root)
            for f in filenames:
                os.makedirs("{0:}/{1:}".format(root, f))
        except IOError:
            raise Exception("cannot create directory: {0:}".format(f))

    def create_directory(root, file):
        if not os.path.exists('{0:}/{1:}'.format(root, file)):
            print(file)
            try:
                os.makedirs("{0:}/{1:}".format(root, file))
            except IOError:
                raise Exception("cannot create directory: {0:}".format(file))

    if not os.path.exists(data_dir):
        create_directory_structure(data_dir, [])
    if not os.path.exists(figure_dir):
        create_directory_structure(figure_dir, figure_list)
    for f in figure_list:
        create_directory(figure_dir, f)

    return data_dir, figure_dir
