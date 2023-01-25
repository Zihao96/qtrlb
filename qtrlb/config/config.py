from copy import deepcopy
from ruamel.yaml import load, dump, CSafeLoader, CSafeDumper, error
from numpy import ndarray
import os
import warnings
warnings.simplefilter('ignore', error.MantissaNoDotYAML1_1Warning)  
# To ignore the message when we load .yaml files. 
# It happens because They require dot for float number


class Config:
    """ This is the parent class of all managers, which loads and stores a 
        YAML config_dict file, provides a dictionary like interface to the 
        config_dict, with the additions of load, save, and variables.
        
        Attributes:
            yamls_path: An absolute path of the directory containing all yamls with a template folder.
            suffix: 'DAC' or 'DataConfig' or 'Variable', etc.
            variable_suffix: '_EJEC' or '_ALGO' or empty string '' for other managers.
            varman: A Variable manager for other manager to load parameters.
    
    """
    def __init__(self, 
                 yamls_path: str, 
                 suffix: str = '', 
                 variable_suffix: str = '',
                 varman = {}):
        self.yamls_path = yamls_path
        self.suffix = suffix
        self.variable_suffix = variable_suffix
        self.varman = varman
        
        self.raw_file_path = os.path.join(self.yamls_path, self.suffix + self.variable_suffix + '.yaml')  
        self.template_file_path = os.path.join(self.yamls_path, 'Template', self.suffix + '.yaml')  
        
        self.checkconfig()
        
        
    def checkconfig(self):
        """
        Check the structure of the yaml file by comparing with its template.
        Raise error if the structure has inconsistency.
        Generate attribute self.config_raw if all check pass.
        
        Notes: We only check Q0 and its '01' and '12' subspace in Variables.yaml.
        """
        # Check the things are actually there.
        try:  
            with open(self.raw_file_path, 'r') as f:
                self.config_raw = load(f, CSafeLoader)
            with open(self.template_file_path, 'r') as f:
                self.config_template = load(f, CSafeLoader)
        except FileNotFoundError as e:
            print('Missing Yaml file or its template. Please check your working directory!!!')
            raise e
            
        # Recursively checking each keys
        self.compare_dict(self.config_raw, self.config_template, suffix=self.suffix+self.variable_suffix)
        
        
    @staticmethod
    def compare_dict(dict_raw: dict, dict_template: dict, suffix: str = ''):
        """
        Recursively checking the keys of two dictionaries.
        Suffix tells which yaml we are checking.
        """
        for k, v in dict_template.items():
            assert k in dict_raw, f'There is no key "{k}" in your {suffix}.yaml'
            if isinstance(v, dict): 
                Config.compare_dict(dict_raw[k], dict_template[k], suffix=suffix)
            
    
    def save(self):
        """
        Save the config_raw dictionary to the yaml file.
        Be careful since the old content will be overwritten!!!
        Notes:
        We shouldn't save config_dict since we only want the change happen at one places.
        The f.write only take string, and dump indeed return to a string.
        """
        config_str = dump(self.config_raw, Dumper=CSafeDumper, default_flow_style=False)
        with open(self.raw_file_path, 'w') as f:
            f.write(config_str)
        print(f'The config_raw of {self.suffix} Manager has been saved successfully.')
        
    
    def load(self):
        """
        Reload the yaml file to config_raw, so that changes in file will be updated.
        Then generate the self.config_dict with replaced variables.
        """
        self.checkconfig()  # To reload the files and prevent accidentally structure change.
        self.config_dict = deepcopy(self.config_raw)
        if self.varman != {}: self.replace_vars(self.config_dict, self.varman)  
        # This is the real difference between config_dict and config_raw
        
        
    @staticmethod
    def replace_vars(dict_to_replace: dict, varman):
        """
        Recursively go deep into a nested dictionary. 
        If the bottom value is a string, try to replace it by the corresponding key in Variable.yaml
        """
        for k, v in dict_to_replace.items():
            
            # If we hit the bottom layer. Try to replace.
            if isinstance(v, str):
                try:
                    dict_to_replace[k] = varman[v]
                except KeyError:
                    pass
                
            # If we haven't reach the bottom layer and v is dict. Go further.
            elif isinstance(v, dict):
                Config.replace_vars(v, varman)
           
                
    ##################################################           
    # All functions below are really used to provide dict-like interface to Config
    # User can access Config.config_dict['SomeKeys'] by simply call Config['SomeKeys']   
    # First, define the native, unchanged method.
    def __iter__(self):
        return self.config_dict.__iter__()

    def __next__(self):
        return self.config_dict.__next__()

    def __repr__(self):
        return self.config_dict.__repr__()

    def __str__(self):
        return self.config_dict.__str__()

    def __len__(self):
        return self.config_dict.__len__()

    def items(self):
        return self.config_dict.items()

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        return self.set(key, value)
    
    def __delitem__(self, key):
        return self.delete(key)
    
              
    # Then, define the fancy method with additional functionality.
    def get(self, key: str, which: str = 'dict'):
        """
        Get the value in the config_dict or config_raw corresponds to a given key.
        Examples:
            varman = Config('.', '_EJEC')
            varman.get("common")
            varman.get("Q0/01/amp_rabi")

        Parameters:
            key: str. String of keys separated by forward slash.
            which : str. Choose which dictionary we look up. Should only be 'dict' or 'raw'.
        """
        keys_list = self.slashed_string_to_list(key)
        
        # Choose which dictionary to look up.
        if which == 'dict':
            result = self.config_dict
        elif which == 'raw':
            result = self.config_raw
        else:
            raise ValueError("Parameter 'which' can only take string 'dict' or 'raw'.")
        
        # Going deep into the dictionary recursively. Detailed exception message is necessary here. 
        for i, key in enumerate(keys_list):
            try:
                result = result[key]
            except KeyError:
                raise KeyError(f'There is no key "{key}" in {"/".join(keys_list[:i])}')

        return result
    
    
    def keys(self, key: str, which: str = 'dict'):
        """
        Get the keys of a dictionary inside the config_dict or config_raw corresponds to a given key.

        Parameters:
            key: str. String of keys separated by forward slash.
            which : str. Choose which dictionary we look up. Should only be 'dict' or 'raw'.
        """
        result = self.get(key, which)
        return result.keys()
    
    
    def set(self, key: str, value, which: str = 'both', save_raw: bool = False):
        """
        Set the value in the config_dict (and config_raw if asked) corresponds to a given key.
        Then save it if asked.
        If there is no corresponding key, a tree of dict will be built.

        Parameters
            key : str. String of keys separated by forward slash.
            value : TYPE. Value we want to set to given key.
            which : str. Choose which dictionary we look up. Should only be 'dict' or 'both'.
            save_raw : bool. Whether we want to save config_raw or not.
        """
        keys_list = self.slashed_string_to_list(key)
        
        # There is a bug in pyyaml load which fails for ndarrays.
        # We can just cast them to a list, which improves readability as well.
        if isinstance(value, ndarray): value = value.tolist()
        
        if which == 'dict':
            self.recursively_set(self.config_dict, keys_list, value)
        elif which == 'both':
            self.recursively_set(self.config_dict, keys_list, value)
            self.recursively_set(self.config_raw, keys_list, value)
            if save_raw: self.save()
        else:
            raise ValueError("Parameter 'which' can only take string 'dict' or 'both'.")
    
    
    def delete(self, key: str, save_raw: bool = False):
        """
        Delete the item in the config_dict and config_raw corresponds to a given key.
        Then save it if asked.

        Parameters
            key : str. String of keys separated by forward slash.
            save_raw : bool. Whether we want to save config_raw or not.
        """
        keys_list = self.slashed_string_to_list(key)
        
        # Get the dict corresponds to the second last key in keys_list.
        base_dict = self.get(''.join(keys_list[:-1]), which='dict')
        base_raw = self.get(''.join(keys_list[:-1]), which='raw')
        
        # Delete the item corresponds to the last key in keys_list.
        del base_dict[keys_list[-1]]
        del base_raw[keys_list[-1]]
        if save_raw: self.save()
        
        
    
    @staticmethod
    def slashed_string_to_list(string: str, remove_empty: bool = True) -> list:
        """
        Generate a list of string based on a slashed string.
        For example: string 'aa/bb/cc/' will return to ['aa', 'bb', 'cc', ''].
        If remove_empty is True, all '' in list will be removed.
        """
        # Make sure the input is indeed a string.
        try:
            result = string.split('/')
        except AttributeError as e:
            print(f'The {string} is not a string.')
            raise e
        
        # Remove all empty string ''            
        if remove_empty:
            for element in result:
                if element == '': result.remove(element)
        
        return result
            
            
    @staticmethod
    def recursively_set(config_dict: dict, keys_list: list, value):
        """
        A helper function for 'set' method. The real process of set happens here.
        
        Here I'm dealing with two situation at same time. KeyError if config_dict miss the key, 
        or AssertionError if it has the key but the value is not a dictionary.
        I bet you can't make it shorter with same functionality. -Zihao(2023/01/24)  
        """
        for i, key in enumerate(keys_list[:-1]):
            try:
                assert isinstance(config_dict[key], dict)
                config_dict = config_dict[key]
            except (KeyError, AssertionError):  
                print(f'A new empty dictionary will be created in {"/".join(keys_list[:i])} with key "{key}".')
                config_dict[key] = {}  
                config_dict = config_dict[key]  # We have to use such pointer movement instead of config_dict={}.  
        config_dict[keys_list[-1]] = value
        
            
    

        
        