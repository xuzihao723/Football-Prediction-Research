import importlib

def test_all_experiment_modules_import():
    for x in 'abcde': importlib.import_module(f'src.experiments.experiment_{x}')
