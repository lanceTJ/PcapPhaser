import configparser

# Function to load configuration from ini file
def load_config(config_path='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_path)
    return {
        'timeout': float(config['pss']['timeout_sec']),
        'short_threshold': int(config['pss']['short_threshold']),
        'long_threshold': int(config['pss']['long_threshold']),
        'batch_size': int(config['pss']['batch_size']),
        'lambda_dict': eval(config['pss']['lambda_dict']),
        'feature_weights': eval(config['pss']['feature_weights']),
        'regularization_lambda': float(config['pss']['regularization_lambda']),
        'max_workers_cfm': int(config['cfm']['max_workers']),
        'timeout_cfm': int(config['cfm']['timeout_min']),
        'max_workers_concat': int(config['concat']['max_workers']),
        'mask_features_concat': [f.strip() for f in config['concat']['mask_features'].split(',')],
        'max_workers_label': int(config['label']['max_workers']),
        'mask_features_label': [f.strip() for f in config['label']['mask_features'].split(',')],
        'rules_file_label': config['label']['rules_file']
    }