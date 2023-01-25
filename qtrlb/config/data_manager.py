import datetime
import os
import pandas as pd
import json
from qtrl.utils import Config
import logging
logger = logging.getLogger('qtrl.DataManager')

# from BlokLabCode import data_saving as ds #TODO: merge these functions into the class -RP
from qtrl.utils.data_utils import save_data_with_config

class DataManager(Config):

    def __init__(self, config_file=None, variables=None, session=None):
        """
        :param config_file: Must contain keys:
                    - (string) base_directory
                    - (string) metadata_directory
                    - (string) data_format
                    - (string) metadata_format
        :param variables: normal config variables argument, no clear use here yet
        :param session: (string) name of experimental session (default today's date YYYY-MM-DD).
                        This will tag the metadata .csv with this session tag
        """
        # standard config initialization
        super().__init__(config_file=config_file, variables=variables)

        # set the directory where data is stored
        self.base_directory = self._config_dict['base_directory']  # where data is saved
        self.metadata_directory = self._config_dict['metadata_directory']  # location of meta-data database
        self.save_directory = None
        self.date_fmt = '%Y%m%d'
        self.time_fmt = '%H%M%S'
        # session is enumerated by the day, but can be customized to be any stringable thing
        if session is None:
            logger.info(f"No session set...setting to {datetime.datetime.now().strftime(self.date_fmt)}")
            self.session = datetime.datetime.now().strftime(self.date_fmt)
        else:
            self.session = session

        # initialize the database for the session
        self.data_format = self._config_dict.get('data_format','.hdf5')
        self.metadata_format = self._config_dict.get('metadata_format','.csv')
        self._metadata_file = os.path.join(self.metadata_directory,f'metadata_{self.session}{self.metadata_format}')

        self.set_exp_id()
        self.update_ID = self._config_dict['update_ID']  # this will update for every call to make_save_dir()
        self.prepend_ID = self._config_dict['prepend_ID']
        # if self.update_ID:
        #     self.set_exp_id()

        self.saved_files = []

    def set_metadata_file(self, session=None):
        if session is None:
            return
        else:
            self.session = session
            self._metadata_file = rf'{self.metadata_directory}/metadata_{self.session}.{self.metadata_format}'

    def get_session_metadata(self, session=None):
        """
        Wrapper to get the meta-data database as a Pandas dataframe
        """
        self.set_metadata_file(session=session)
        assert os.path.exists(self._metadata_file), f"No database exists at {self._metadata_file}"
        with open(self._metadata_file, 'r') as f:
            df = pd.read_csv(f)
        return df

    def make_query_string(self, query):
        """
        Simple formatter for making DataFrame queries.
        :param query: tuple, (column, boolean, entry
        :return:      formatted query string to input into df.query()
        """
        query_string = f'({query[0]} {query[1]} {query[2]})'
        return query_string

    def get_metadata_by_field(self, field=None, session=None):
        """
        Get metadata for all experiments with a given field in experiment session
        :param field:      tuple, field[0] is key, field[1] is value
        :param session: str, name of the experiment session e.g. '2019_11_13'
        :return:        DataFrame, metadata
        """

        mdf = self.get_session_metadata(session=session)
        if field is not None:
            column, entry = field
            mdf_with_field = mdf.query(f'{column} == {entry}')
        else:
            mdf_with_id = mdf
        return mdf_with_id

    def get_metadata_by_id(self, exp_id=None, session=None):
        """
        Get metadata for all experiments with a given id in experiment session
        :param exp_id:      int, experiment-id
        :param session: str, name of the experiment session e.g. '20191113'
        :return:        DataFrame, metadata
        """
        mdf = self.get_session_metadata(session=session)
        if exp_id is not None:
            mdf_with_id = mdf[mdf['id'] == exp_id]
        else:
            mdf_with_id = mdf
        return mdf_with_id



    def load_data_from_metadata(self, metadata_df):
        data_df = []
        for filepath in metadata_df['path']:
            data_df.append(pd.read_json(filepath))

        data_df = pd.concat(data_df, ignore_index=True)
        return data_df

    def load_data_from_paths(self, paths):
        data_df = []
        for filepath in paths:
            data_df.append(pd.read_json(filepath))

        data_df = pd.concat(data_df, ignore_index=True)
        return data_df

    def make_save_dir(self, dir_string='', date='auto', date_first=True, add_session_directory=False, add_timestamp=True, experiment_suffix=''):
        """
        Makes a save_directory for the data given directory string. By default
        the order is
        <base_dir> / <date> / <dir_string>,

        but, if date is False:
        <base_dir> / <dir_string>

        if date_first=True:
        <base_dir> /<dir_string> / <date>,

        depending on preference

        Returns save_dir
        """
        # Make a new unique identifier when making a new directory for data to be stored
        if self.update_ID:
            self.set_exp_id()

        # add the ID to the subdirectory, if prepend_ID is true
        if dir_string != '' and self.prepend_ID:
            dir_string = rf'{self.exp_ID}_{dir_string}'
            
        # add the date automatically
        if date == 'auto':
            time_str = self.stamp_time(self.time_fmt)
            date_str = self.stamp_time(self.date_fmt)
        elif type(date) is datetime.datetime:
            time_str = date.strftime(self.time_fmt)
            date_str = date.strftime(self.date_fmt)
        elif date is None or date=='none':
            time_str = ''
            date_str = ''

        if add_timestamp:
            dir_string = time_str+'_'+dir_string



        if add_session_directory:
            session_directory = os.path.join(self.base_directory,f'session_{self.session}')
        else:
            session_directory = self.base_directory
        # decide ordering which to put the date
        if date_first:
            save_dir = os.path.join(session_directory,date_str,dir_string).rstrip(os.sep)
        else:
            save_dir = os.path.join(session_directory,dir_string,date_str).rstrip(os.sep) #strip is just in case we end up with trailing slashes

        if experiment_suffix != '':
            save_dir = save_dir + '_' + experiment_suffix


        # or don't add the date
        # if date is None:  # if you want to make a directory that doesn't specify date, for saving more general stuff
        #     save_dir = os.path.join(session_directory,dir_string)
        #Commented out and moved upwards -RP (8/24/21)

        if not os.path.exists(save_dir):
            logger.info("Data storage dir does not exist,"
                        " creating it at {}".format(save_dir))
            os.makedirs(save_dir)

        return save_dir

    def stamp_time(self, fmt=None):
        return datetime.datetime.now() if fmt is None else datetime.datetime.now().strftime(fmt)

    def _filename_timestamp(self, timestamp=None, filename='noName'):
        if timestamp is None or timestamp=='auto':
            filename = self.stamp_time(self.time_fmt) + '_' + filename
        elif type(timestamp) is datetime.datetime:
            filename = timestamp.strftime(self.time_fmt) + '_' + filename
        elif type(timestamp) is str:
            filename = timestamp + '_' + filename
        return filename


    def _append_metadata_to_database(self, metadata):
        """
        Appends (dict-like) metadata to the session metadata database.

        """
        df = pd.DataFrame([metadata])
        if os.path.exists(self._metadata_file):
            session_df = self.get_session_metadata()
            df = pd.concat([session_df, df], axis=0, ignore_index=True, sort=True)

        with open(self._metadata_file, 'w') as f:
            df.to_csv(f, index=False)

    def set_exp_id(self):
        self.exp_ID = int(datetime.datetime.timestamp(datetime.datetime.now()))

    def save_data(self, dir_string, filename, data_dict, config_dict, experiment_kwargs={}, date='auto', extension='', experiment_suffix=''):

        if not extension.startswith('.'):
            extension = '.'+extension
        self.save_directory = self.make_save_dir(dir_string=dir_string, date=date, add_timestamp=True, experiment_suffix=experiment_suffix)
        
        print('saving to',self.save_directory, flush=True)
        
        filepath = os.path.join(self.save_directory, 
                                self._filename_timestamp(timestamp=date, 
                                                         filename=filename))
        if experiment_suffix != '':
            filepath = filepath + '_' + experiment_suffix
        
        filepath = filepath + extension
        
        file_metadata = {'path': filepath,
                          'id': self.exp_ID,
                          'timestamp': f'{datetime.datetime.now()}'}
        metadata = {**file_metadata, **config_dict}
        # print(f'metadata keys: {metadata.keys()}')
        # self._append_metadata_to_database(file_metadata)
        if 'config' not in data_dict:
            data_dict['config'] = metadata

        if extension.lower() == '.json':
            data_df = pd.DataFrame([data_dict])
            if os.path.exists(filepath):
                logger.info('File exists! not saving...')
            else:
                with open(filepath, 'w') as f:
                    data_df.to_json(f)
                # add to quick-access list of saved
                self.saved_files.append(file_metadata)
        elif extension.lower() == '.hdf5' or extension == '':
            save_data_with_config(data_dict, filepath, metadata=metadata, sequence_settings=experiment_kwargs)
        else:
            logger.info('Need to implement other formats besides .json and .hdf5!')
        return filepath

