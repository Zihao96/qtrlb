





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
                 readout_resonators,
                 subspace='01',
                 prepulse=None,
                 postpulse=None,
                 fitmodel=None):
        self.config = config
        self.scan_name = scan_name
        self.x_label = x_label
        self.x_unit = x_unit
        self.scan_start = scan_start
        self.scan_stop = scan_stop
        self.npoints = npoints
        self.drive_qubits = self.make_it_list(drive_qubits)
        self.readout_resonators = self.make_it_list(readout_resonators)
        self.subspace = subspace
        self.prepulse = self.make_it_list(prepulse)
        self.postpulse = self.make_it_list(prepulse)
        self.fitmodel = fitmodel
        
        
    def run(self):
        self.initialize()
        self.make_sequence()  #TODO: It should generate the sequence program.
        #TODO: Should we separate it into make and then compile
        self.upload_sequence()
        self.acquire_data()  # This is really run the thing and return to the IQ data.
        self.analyze_data()
        self.plot()
        
        
    def initialize(self):
        """
        Configure the Qblox based on drive_qubits and readout_resonators using in this scan.
        Warn user if any drive_qubits are not being readout without raising error.
        We call these methods here instead of during init/load of DACManager,
        because we want those modules/sequencers not being used to keep their default status.
        """
        for qubit in self.drive_qubits:
            if qubit not in self.readout_qubits:
                print(f'The Q{qubit} will not be readout!')
                
        self.config.DAC.implement_parameters(qubits=self.drive_qubits, 
                                             resonators=self.readout_resonators,
                                             subspace=self.subspace)
        
        
    def make_sequence(self):
        """
        Generate self.sequence, which should be a instance of Sequence class.
        Inside this class, we expect to have each sequence_dict mapping to a physical sequencer,
        and thus mapping to each drive/readout qubits.
        """
        raise NotImplementedError("This method should be implemented in its child class.")
        
        
    @staticmethod
    def make_it_list(thing):
        """
        A crucial, life-saving function.
        """
        if isinstance(thing, list):
            return thing
        elif thing == None:
            return []
        else:
            return [thing]

















































