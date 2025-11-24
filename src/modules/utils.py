import configparser
import ast  # Import for safe literal evaluation

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
    return {
        'timeout_sec': float(config['pss'].get('timeout_sec', '600')),
        'short_threshold': int(config['pss'].get('short_threshold', '10')),
        'long_threshold': int(config['pss'].get('long_threshold', '100')),
        'batch_size': int(config['pss'].get('batch_size', '100')),
        'lambda_dict': ast.literal_eval(lambda_dict_str),
        'feature_weights': ast.literal_eval(feature_weights_str),
        'regularization_lambda': float(config['pss'].get('regularization_lambda', '1e-3')),
        'max_workers_cfm': int(config['cfm'].get('max_workers', '8')),
        'timeout_min_cfm': int(config['cfm'].get('timeout_min', '60')),
        'max_workers_concat': int(config['concat'].get('max_workers', '8')),
        'mask_features_concat': [f.strip() for f in mask_features_concat_str.split(',')],
        'max_workers_label': int(config['label'].get('max_workers', '8')),
        'mask_features_label': [f.strip() for f in mask_features_label_str.split(',')],
        'rules_file': config['label'].get('rules_file', 'cic_improved_2018_rules.yaml'),
        'allowed_feature_names': allowed_feature_names,  # list
        'num_phases_list': num_phases_list,  # list of int
        'max_flow_length': int(config['pss'].get('max_flow_length', '1000')),
        'min_flow_length': int(config['pss'].get('min_flow_length', '3'))
    }