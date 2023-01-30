





class Scan:
    """ Base class for all parameter-sweep experiment.
        The framework of how experiment flow will be constructed here.
        It should be used as parent class of specific scan rather than be instantiated directly.
        
        Attributes:
            config: A MetaManager.
    """
    def __init__(self, 
                 config, 
                 scan_name,
                 x_label, 
                 x_unit, 
                 scan_start, 
                 scan_stop, 
                 npoints, 
                 drive_qubits,
                 readout_qubits,
                 subspace,
                 prepulse=None,
                 postpulse=None,
                 fitmodel=None,
                 level_to_fit='from',
                 two_level=False):
        self.config = config
        self.scan_name = scan_name
        self.x_label = x_label
        self.x_unit = x_unit
        self.scan_start = scan_start
        self.scan_stop = scan_stop
        self.npoints = npoints
        self.drive_qubits = drive_qubits
        self.readout_qubits = readout_qubits
        self.subspace = subspace
        self.prepulse = prepulse
        self.postpulse = postpulse
        
        
    def run(self):
        self.initialize()  #TODO: It should determine how many qubits to drive and readout and assign the sequencer.
        self.make_sequence()  #TODO: It should generate the sequence program.
        #TODO: Should we separate it into make and then compile
        self.upload_sequence()
        self.acquire_data()  # This is really run the thing and return to the IQ data.
        self.analyze_data()
        self.plot()
        
        
    def initialize(self):
        self.cfg.DAC.reset()
        self.map_readout_sequencers()


    def make_sequence(self):
        """
        Generate self.sequence, which should be a instance of Sequence class.
        Inside this class, we expect to have each sequence_dict mapping to a physical sequencer,
        and thus mapping to each drive/readout qubits.
        """
        raise NotImplementedError("This method should be implemented in its child class.")
        

    def map_readout_sequencers(self):
        """
        Map each readout_qubits to a sequencer based on their set output port.
        This is virtual map, and the real 'channel_map' command is called in DACManager. 
        """
        self.readout_sequencers_map = {}























































