import configparser
import ast  # Import for safe literal evaluation
import os

# Function to load configuration from ini file
def load_config(config_path='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_path)
    allowed_feature_names_str = config['pss'].get('allowed_feature_names', 'packet_length,inter_arrival_time,direction,up_down_rate')
    allowed_feature_names = [ft.strip() for ft in allowed_feature_names_str.split(',')]
    lambda_dict_str = config['pss'].get('lambda_dict', "{'packet_length': 1e-3, 'inter_arrival_time': 1e-3, 'up_down_rate': 1e-3, 'direction': 1e-3}")
    feature_weights_str = config['pss'].get('feature_weights', "{'packet_length': 0.5, 'inter_arrival_time': 0.3, 'up_down_rate': 0.1, 'direction': 0.1}")
    num_phases_str = config['pss'].get('num_phases', '2,3,4').split(';')[0].strip()  # Clean comment
    num_phases_list = [int(p) for p in num_phases_str.split(',')]
    mask_features_concat_str = config['concat'].get('mask_features', 'Flow ID')
    mask_features_label_str = config['label'].get('mask_features', 'Flow ID')

    # Resolve absolute path for rules_file
    rules_file_path = config['label'].get('rules_file', '../rules_file/cic_improved_2018_rules.yaml')
    config_dir = os.path.dirname(config_path)
    absolute_rules_file_path = os.path.normpath(os.path.join(config_dir, rules_file_path))
    config['label']['rules_file'] = absolute_rules_file_path
    return {
        'pss': {
            'allowed_feature_names': allowed_feature_names,
            'num_phases': num_phases_list,
            'max_flow_length': config['pss'].getint('max_flow_length', 200),
            'min_flow_length': config['pss'].getint('min_flow_length', 3),
            'timeout_sec': config['pss'].getint('timeout_sec', 600),
            'lambda_dict': ast.literal_eval(lambda_dict_str),
            'feature_weights': ast.literal_eval(feature_weights_str),
            'regularization_lambda': config['pss'].getfloat('regularization_lambda', 1e-3),
            'short_threshold': config['pss'].getint('short_threshold', 10),
            'long_threshold': config['pss'].getint('long_threshold', 100),
            'batch_size': config['pss'].getint('batch_size', 100),
        },
        'cfm': {
            'max_workers': config['cfm'].getint('max_workers', 8),
            'timeout_min': config['cfm'].getint('timeout_min', 60),
        },
        'concat': {
            'max_workers': config['concat'].getint('max_workers', 8),
            'mask_features': [mf.strip() for mf in mask_features_concat_str.split(',')],
        },
        'label': {
            'max_workers': config['label'].getint('max_workers', 8),
            'mask_features': [mf.strip() for mf in mask_features_label_str.split(',')],
            'rules_file': config['label'].get('rules_file', '../rules_file/cic_improved_2018_rules.yaml'),
            'time_zone_adjustment': config['label'].getboolean('time_zone_adjustment', False),
            'time_format': config['label'].get('time_format', '%d/%m/%Y %I:%M:%S %p'),
        }
    }