import yaml

def load_model_config(model, mode):
    # load special config for each model
    config_path = f'/home/user/git_code/LMOE/config/{model}.yaml'
    print(f'[!] load configuration from {config_path}')
    with open(config_path) as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
        new_config = {}
        for key, value in configuration.items():
            if key in ['train', 'test', 'validation']:
                if mode == key:
                    new_config.update(value)
            else:
                new_config[key] = value
        configuration = new_config
    return configuration

def load_config(args):
    '''the configuration of each model can rewrite the base configuration'''
    # base config
    base_configuration = load_base_config()
    # print("=============base_configuration will be printed===============")
    # print(base_configuration)

    # load one model config
    configuration = load_model_config(args['model'], args['mode'])

    # update and append the special config for base config
    base_configuration.update(configuration)
    # print("=============base_configuration will be changed!===============")
    # print(base_configuration)
    configuration = base_configuration

    return configuration

def load_base_config():
    config_path = f'/home/user/git_code/LMOE/config/base.yaml'
    with open(config_path) as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    print(f'[!] load base configuration: {config_path}')
    return configuration
