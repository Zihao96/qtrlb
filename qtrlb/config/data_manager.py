import os
import glob
import h5py
import datetime
import traceback
import numpy as np
from qtrlb.config.config import Config
from qtrlb.config.variable_manager import VariableManager


class DataManager(Config):
    """ This is a thin wrapper over the Config class to help with data management.
        The load() method will be called once in its __init__.
        This manager can be used independent of VariableManager by pass in varman=None.
        It will be useful for experiment outside of qtrl.calibration framework.
    
        Attributes:
            yamls_path: An absolute path of the directory containing all yamls with a template folder.
            varman: A VariableManager.
    """
    def __init__(self, 
                 yamls_path: str, 
                 varman: VariableManager = None):
        super().__init__(yamls_path=yamls_path, 
                         suffix='data',
                         varman=varman)
        self.load()


    def make_exp_dir(self, experiment_type: str, experiment_suffix: str, time: datetime.datetime = None):
        """
        Make a saving directory under self['base_directory'].
        Return to path: 'base_directory/date_fmt/time_fmt+experiment_type+experiment_suffix'.
        Notice the basename won't repeat here.
        Other function can get basename by calling os.path.basename(data_path)
        Overwriting is forbidden here.
        
        Example of Attribute:
            experiment_type: 'drive_amplitude', 't1', 'ramsey'
            experiment_suffix: 'LLR2_EJEC_Q4_AD+200kHz_happynewyear'
        """
        _chip_name = self.varman.variable_suffix if self.varman is not None else ''
        
        self.datetime = time if time is not None else datetime.datetime.now()
        date = self.datetime.strftime(self['date_fmt'])
        time = self.datetime.strftime(self['time_fmt'])
        
        self.datetime_stamp = '/'.join([date, time])
        self.basename = time + _chip_name + '_' + experiment_type + '_' + experiment_suffix
        
        self.data_path = os.path.join(self['base_directory'], date, self.basename)
        self.yamls_path = os.path.join(self.data_path, 'Yamls')
        self.jsons_path = os.path.join(self.data_path, 'Jsons')
        
        try:
            os.makedirs(self.yamls_path)
            os.makedirs(self.jsons_path)
        except FileExistsError:
            print('DataManager: Experiment directory exists. No directory will be created.')
            print(traceback.format_exc())


    @staticmethod
    def save_measurement(data_path: str, measurement: dict, attrs: dict = None):
        """
        Save the measurement dictionary into a hdf5.
        I keep this layer because we can also pass information of scan here and save it as attrs.
        Some attributes, cfg for instance, cannot (and shouldn't) be saved. They raise TypeError.
        Attributes about gates may be stringized.
        """
        if attrs is None: attrs = {}
        hdf5_path = os.path.join(data_path, 'measurement.hdf5')  
        
        with h5py.File(hdf5_path, 'w') as h5file:
            DataManager.save_dict_to_hdf5(measurement, h5file)
            for k, v in attrs.items(): 
                try:
                    h5file.attrs[k] = v
                except TypeError:  
                    # If v is a dictionary(pre_gate, process_kwargs), it will also raise TypeError.
                    if isinstance(k, str) and k.split('_')[-1] in ('gate', 'kwargs'):
                        h5file.attrs[k] = str(v)
                    else:
                        pass
                except Exception:
                    pass


    @staticmethod
    def load_measurement(data_path: str) -> tuple[dict, dict]:
        """
        Load the measurement dictionary from a hdf5 file.
        Return measurement dictionary and attributes dictionary.
        data_path should be the folder path without the measurement.hdf5 at end.
        """
        hdf5_path = os.path.join(data_path, 'measurement.hdf5')

        with h5py.File(hdf5_path, 'r') as h5file:
            measurement = DataManager.load_hdf5_to_dict(h5file)
            attrs = {k: v for k, v in h5file.attrs.items()}

        return measurement, attrs
        
        
    @staticmethod
    def save_dict_to_hdf5(dictionary: dict, h5: h5py.File | h5py.Group):
        """
        Recursively save a nested dictionary to hdf5 file/group. 
        """
        for k, v in dictionary.items():
            if isinstance(v, dict):
                subgroup = h5.require_group(k)
                DataManager.save_dict_to_hdf5(v, subgroup)
            elif v is None:
                continue
            else:
                h5.create_dataset(k, data = v)

                
    @staticmethod                
    def load_hdf5_to_dict(h5: h5py.File | h5py.Group, dictionary: dict = None) -> dict:
        """
        Recursively load a hdf5 file/group to a nested dictionary.
        """
        if dictionary == None: dictionary={}
        dictionary = dict(h5)
        
        for k, v in dictionary.items():
            if isinstance(v, h5py.Group):
                dictionary[k] = DataManager.load_hdf5_to_dict(v, dictionary[k])
            else:
                dictionary[k] = np.array(v)
                
        return dictionary
    

    @staticmethod
    def get_data_paths(datetime_stamp_start: str, datetime_stamp_stop: str, base_directory: str) -> list:
        """
        Extract all the data directories (those include measurement.hdf5) within a given time interval. 
        Return to a list whose elements are string of paths in time order.
        The speed of this function usually limited by the speed of loading the folder structure from remote Box server.

        Parameters
        ----------
        datetime_stamp_start : Example: '20221115/235852'. Forward slash is fine  here for Windows.
        datetime_stamp_stop : Example: '20221116/000051'
        base_directory : The path of the data folder until the 'Qblox'. Typically cfg['data.base_directory'].
                         Might need to be changed based on OS and username.
        """
        data_paths = []
        first_date, first_time = os.path.split(datetime_stamp_start)
        last_date, last_time = os.path.split(datetime_stamp_stop)

        if last_date < first_date:
            raise ValueError('The datetime_stamp_stop should be later than datetime_stamp_start!!!')
            
        elif last_date == first_date:
            if last_time < first_time:
                raise ValueError('The datetime_stamp_stop should be later than datetime_stamp_start!!!')

            # Add data_path in this date between first and last time
            data_paths.extend(
                [
                    data_path for data_path in glob.glob(os.path.join(base_directory, first_date, '*')) 
                    if (os.path.basename(data_path)[:6] >= first_time 
                        and os.path.basename(data_path)[:6] <= last_time) 
                ]
            )
        
        elif last_date > first_date:
            # Add data_path in first date
            data_paths.extend(
                [
                    data_path for data_path in glob.glob(os.path.join(base_directory, first_date, '*')) 
                    if os.path.basename(data_path)[:6] >= first_time
                ]
            )

            # Add data_path in last date
            data_paths.extend(
                [
                    data_path for data_path in glob.glob(os.path.join(base_directory, last_date, '*')) 
                    if os.path.basename(data_path)[:6] <= last_time
                ]
            )

            # If there are more than 2 days
            first_date, last_date = (int(first_date), int(last_date))
            if last_date-1 != first_date:
                for date in range(first_date+1, last_date):
                    data_paths.extend(
                        [data_path for data_path in glob.glob(os.path.join(base_directory, str(date), '*'))]
                    )

        return sorted(data_paths)


    @staticmethod
    def get_time_lengths(datetime_stamp_start: str, datetime_stamp_stop: str, 
                         time_points: int, time_unit: str = 'hrs') -> np.ndarray:
        """
        Get a numpy array of equally spaced relative time values based on start, stop time and points.
        Only days, seconds and microseconds are stored internally in datetime.timedelta object.
        It's very useful for making plot of overnight scan.

        Example:
        datetime_stamp_start = '20230716/203000'
        datetime_stamp_stop = '20230717/153000'
        Return to [0, 1, 2, ..., 19]
        """
        datetime_start = datetime.datetime.strptime(datetime_stamp_start, '%Y%m%d/%H%M%S')
        datetime_stop = datetime.datetime.strptime(datetime_stamp_stop, '%Y%m%d/%H%M%S')
        datetime_delta = datetime_stop - datetime_start
        total_time_sec = (datetime_delta.days * 24 * 3600 
                        + datetime_delta.seconds 
                        + datetime_delta.microseconds * 1e-6)

        if time_unit.lower().startswith('s'):
            total_time = total_time_sec
        elif time_unit.lower().startswith('min'):
            total_time = total_time_sec / 60
        elif time_unit.lower().startswith('h'):
            total_time = total_time_sec / 3600
        else:
            raise ValueError(f"DataManager: Doesn't support time_unit [{time_unit}] yet")

        return np.linspace(0, total_time, time_points)