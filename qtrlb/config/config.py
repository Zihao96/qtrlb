from copy import deepcopy
from ruamel.yaml import load, dump, CSafeLoader, CSafeDumper, error
from numpy import ndarray
import warnings
warnings.simplefilter('ignore', error.MantissaNoDotYAML1_1Warning)  
# To ignore the message when we load .yaml files. 
# It happens because They require dot for float number


# This class exists to provide a unique label so that we can use None as an optional input for the get function
class NoSetting:
    pass


class Config:
    """Loads and stores a YAML _config_dict file, provides a dictionary like
    interface to the _config_dict, with the additions of load, save, and variables"""
    def __init__(self, config_file=None, config_dict=None, variables=None):
        """
        We usually call this class by begin_measurement_session in config_functions.py.
        For VariableManager, we pass the path string of variable.yaml to argument 'config_file'.
        For all other Manager, we pass their own .yaml to config_file along with the VariableManager to argument 'variables'.
        --Z
        """

        if variables is None:
            self._variables = {}
        else:
            self._variables = variables  # VariableManager
        self._config_dict = config_dict  # dict
        self._config_raw = config_dict   # dict
        self._config_file = config_file  # str
        self.load()

    def load(self, config_raw=None):
        """
        This function really load things in the .yaml file and assign it to _config_raw and _config_dict.
        And then do the trick of replacing variable.
        """
        # Try to open the _config_dict file, if not found, create it and warn the user
        if config_raw is None:
            try:
                with open(self._config_file, 'r') as f:
                    self._config_raw = load(f, CSafeLoader) or {}
            except FileNotFoundError:
                warnings.warn("Config file not found, creating it at {}".format(self._config_file))
                with open(self._config_file, 'w') as _:
                    pass
                self.load()
                return
        else:
            self._config_raw = deepcopy(config_raw)
    
        self._config_dict = deepcopy(self._config_raw)
        
        if self._variables != {}:
            self.replace_vars(self._config_dict, self._variables)  # Can let VariableManager skip this line.
            # This is actually the real big difference between _config_dict and _config_raw. --Z


    def get(self, key, default=NoSetting, reload_config=False):
        """Get a key from the _config_dict, the _config_dict can be traversed with '/'
        a default response can be given, and if the key not found will be returned.
        Otherwise this raises a KeyError if the key is not present.
        
        Examples:
            conf = Config("Example.yaml")
            conf.get("description")
            conf.get("Alazar/Key_not_here", default=0)
            conf.get("Hardware/ADCS/Some/Deep/Value")
        
        Will return to the value corresponds to the deepest key. 
        Or 'default' argument if we cannot find such key. As long as it's not NoSetting
        --Z
        
        """
        if reload_config:
            self.load()

        try:
            keys = key.split('/')  # List of key in each layer.
        except AttributeError:
            raise KeyError()
        if keys[-1] == '':  # Drop last void
            keys = keys[:-1]

        result = self._config_dict
        for i, key in enumerate(keys): #recursion in a for loop
            try:
                result = result[key]
            except KeyError:
                if default != NoSetting:
                    return default
                else:        
                    raise KeyError('/'.join(keys[i:]))
            except TypeError: #Will be raised if the path given is too long -RP (9/2/21)
                raise KeyError('/'.join(keys[i:]))

        return result
    
    def get_raw(self, key, default=NoSetting, reload_config=False):
        """Get a key from the _config_raw, the _config_dict can be traversed with '/'
        a default response can be given, and if the key not found will be returned.
        Otherwise this raises a KeyError if the key is not present.
        
        Examples:
            conf = Config("Example.yaml")
            conf.get("description")
            conf.get("Alazar/Key_not_here", default=0)
            conf.get("Hardware/ADCS/Some/Deep/Value")
        
        """
        if reload_config:
            self.load()

        try:
            keys = key.split('/')
        except AttributeError:
            raise KeyError()
        if keys[-1] == '':
            keys = keys[:-1]

        result = self._config_raw
        for i, key in enumerate(keys): #recursion in a for loop
            try:
                result = result[key]
            except KeyError:
                if default != NoSetting:
                    return default
                else:
                    raise KeyError('/'.join(keys[i:]))
            except TypeError: #Will be raised if the path given is too long -RP (9/2/21)
                raise KeyError('/'.join(keys[i:]))

        return deepcopy(result)


    def save(self):
        conf_str = dump(self._config_raw, Dumper=CSafeDumper, default_flow_style=False)
        # We shouldn't dump _config_dict since we need to change things in multiple place in that case.
        with open(self._config_file, 'w') as f:
            f.write(conf_str)
    
    def set(self, key, value, reload_config=False, save_config=False):
        """Set values in the _config_dict, can be traversed with '/'"""
        if reload_config:
            self.load()

        try:
            keys = key.split('/')
        except AttributeError:
            raise KeyError()

        if keys[-1] == '':
            keys = keys[:-1]

        # We need to traverse both the raw and variable replaced 
        # _config_dict dictionary tree, and replace both simultaneously
        config = self._config_dict
        config_raw = self._config_raw
        for key in keys[:-1]:
            try:
                if not isinstance(config[key], dict):
                    config[key] = {}
                config = config[key]
            except KeyError:  # build the tree if it isn't present
                config[key] = {}
                config = config[key]
        
            try:
                if not isinstance(config_raw[key], dict):
                    config_raw[key] = {}
                config_raw = config_raw[key]
            except KeyError:  # build the tree if it isn't present
                config_raw[key] = {}
                config_raw = config_raw[key]

        # there is a bug in pyyaml load which fails for ndarrays
        # we can just cast them to a list, which improves readability as well
        if isinstance(value, ndarray):
            value = value.tolist()

        # finally update the value
        config[keys[-1]] = value
        config_raw[keys[-1]] = value

        replace_vars(self._config_dict, self._variables)

        if save_config:
            self.save()

    def set_dict(self, key, value, reload_config=False):
        """Set values in the _config_dict, but not the _config_raw, can be traversed with '/'"""
        if reload_config:
            self.load()

        try:
            keys = key.split('/')
        except AttributeError:
            raise KeyError()

        if keys[-1] == '':
            keys = keys[:-1]

        # We will traverse only the replaced 
        # _config_dict dictionary tree, and replace
        config = self._config_dict
        for key in keys[:-1]:
            try:
                if not isinstance(config[key], dict):
                    config[key] = {}
                config = config[key]
            except KeyError:  # build the tree if it isn't present
                config[key] = {}
                config = config[key]
        
        # there is a bug in pyyaml load which fails for ndarrays
        # we can just cast them to a list, which improves readability as well
        if isinstance(value, ndarray):
            value = value.tolist()

        # finally update the value
        config[keys[-1]] = value

        replace_vars(self._config_dict, self._variables)




    def keys(self, key='', reload_config=False):
        """Get the keys at the given location, can be traversed with '/'"""
        if reload_config:
            self.load()

        try:
            keys = key.split('/')
        except AttributeError:
            raise KeyError()

        if keys[-1] == '':
            keys = keys[:-1]

        result = self._config_dict
        for i, key in enumerate(keys):
            try:
                result = result[key]
            except KeyError:
                    raise KeyError('/'.join(keys[i:]))

        return result.keys()

    def __delitem__(self, key):
        """Delete values in the Config, can be traversed with '/'"""
        self.load()

        try:
            keys = key.split('/')
        except AttributeError:
            raise KeyError()

        if keys[-1] == '':
            keys = keys[:-1]

        # We need to traverse both the raw and variable replaced
        # _config_dict dictionary tree, and replace both simultaneously
        config = self._config_dict
        config_raw = self._config_raw
        for key in keys[:-1]:
            config = config[key]

            config_raw = config_raw[key]

        # finally update the value
        del config[keys[-1]]
        del config_raw[keys[-1]]

        self.save()

    # All function below reference the _config_dict as though the Config is this dictionary
    def __iter__(self):
        return self._config_dict.__iter__()

    def __next__(self):
        return self._config_dict.__next__()

    def __repr__(self):
        return self._config_dict.__repr__()

    def __str__(self):
        return self._config_dict.__str__()

    def __len__(self):
        return len(self._config_dict)

    def items(self):
        return self._config_dict.items()

    def __getitem__(self, key):
        return self.get(key, reload_config=False)

    def __setitem__(self, key, item):

        return self.set(key, item)


    @staticmethod
    def replace_vars(dic, variables, max_depth=10, depth=0):
        """Recursively traverse a dictionary and convert any string which
        is in the variable dictionary into the value in the variable dictionary"""
        if depth > max_depth:
            return
        depth += 1
    
        # 2 cases we care about, if we received a dictionary
        if isinstance(dic, dict):
            for key in dic:
                if isinstance(dic[key], dict):
                    replace_vars(dic[key], variables, max_depth=max_depth, depth=depth)
                elif isinstance(dic[key], list):
                    replace_vars(dic[key], variables, max_depth=max_depth, depth=depth)
                else:
                    try:
                        dic[key] = variables[dic[key]]
                    except KeyError:
                        replace_vars(dic[key], variables, max_depth=max_depth, depth=depth)
        # or a list, which we need to iterate over
        elif isinstance(dic, list):
            for i, item in enumerate(dic):
                if not isinstance(item, collections.Hashable):
                    replace_vars(item, variables, max_depth=max_depth, depth=depth)
                else:
                    try:
                        dic[i] = variables[item]
                    except KeyError:
                        replace_vars(item, variables, max_depth=max_depth, depth=depth)