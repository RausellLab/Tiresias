from src import config
from src.utils.misc import product_dict


def param_combinations(stage, model_name):
    for params in product_dict(**config.parameters[stage][model_name]):
        yield params
