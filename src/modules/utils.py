import configparser

# Function to load configuration from ini file
def load_config(config_path='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_path)
    return {
        'timeout': float(config['pss']['timeout']),
        'short_threshold': int(config['pss']['short_threshold']),
        'long_threshold': int(config['pss']['long_threshold']),
        'batch_size': int(config['pss']['batch_size'])
    }