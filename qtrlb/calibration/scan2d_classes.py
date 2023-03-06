from lmfit import Model
from qtrlb.calibration.calibration import Scan2D
from qtrlb.calibration.scan_classes import RabiScan




class ChevronScan(Scan2D, RabiScan):
    def __init__(self,
                 cfg, 
                 drive_qubits: str | list,
                 readout_resonators: str | list,
                 freq_start: float, 
                 freq_stop: float, 
                 freq_points: int, 
                 length_start: float = 0,
                 length_stop: float = 320e-9,
                 length_points: int = 81,
                 subspace: str = None,
                 prepulse: dict = None,
                 postpulse: dict = None,
                 n_seqloops: int = 10,
                 level_to_fit: int | list = None,
                 fitmodel: Model = None,
                 init_waveform_idx: int = 11):
        
        super().__init__(cfg=cfg, 
                         drive_qubits=drive_qubits,
                         readout_resonators=readout_resonators,
                         scan_name='Chevron',
                         x_label_plot='Frequency', 
                         x_unit_plot='[GHz]', 
                         x_start=freq_start, 
                         x_stop=freq_stop, 
                         x_points=freq_points, 
                         y_label_plot='Pulse Length',
                         y_unit_plot='[ns]',
                         y_start=length_start,
                         y_stop=length_stop,
                         y_points=length_points,
                         subspace=subspace,
                         prepulse=prepulse,
                         postpulse=postpulse,
                         n_seqloops=n_seqloops,
                         level_to_fit=level_to_fit,
                         fitmodel=fitmodel)
        
        self.init_waveform_index = init_waveform_idx
        
        
    def add_yinit(self):
        pass
        # TODO: move the set_freq and set_awg_gain of RabiScan out of its add_main so we can reuse it.