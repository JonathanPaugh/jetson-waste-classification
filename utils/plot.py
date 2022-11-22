from os import mkdir
from os.path import join
import configs.model as config


def save_plot(fig, file_name):
    try:
        mkdir(config.OUTPUT_PATH)
    except FileExistsError:
        pass

    fig_path = join(config.OUTPUT_PATH, file_name)
    fig.savefig(fig_path, bbox_inches='tight')
    print(f'Wrote plot to {fig_path}')
