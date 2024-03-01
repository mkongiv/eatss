from init import init_config
from plot_eat import *

def main():
    data_dir, fig_dir = init_config()
    plot_eat(data_dir, fig_dir)

if __name__ == '__main__':
    main()
