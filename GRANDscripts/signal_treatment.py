import numpy as np
from scipy.signal import butter, lfilter, resample
from scipy.fftpack import rfft, irfft, rfftfreq

__all__ = ["add_noise", "digitization", "filter"]

##########################################################################
##########################################################################


def add_noise(vrms, voltages):
    """Add normal random noise on voltages

    inputs : (voltage noise rms, voltages)
    outputs : noisy voltages
    """
    voltages[:, 1:] = voltages[:, 1:] + \
        np.random.normal(0, vrms, size=np.shape(voltages[:, 1:]))
    return voltages


def digitization(voltages, tsampling):
    """Digitize the voltages at an specific sampling

    inputs : (voltages, sampling rate)
    outputs : digitized voltages
    """
    dt = voltages[0, 1] - voltages[0, 0]
    num = len(voltages[0, :]) * int(tsampling / dt)
    if tsampling % dt != 0:
        raise ValueError("Sampling time not multiple of simulation time")
    t_sampled = resample(voltages[0, :], num)
    Vx_sampled = resample(voltages[1, :], num)
    Vy_sampled = resample(voltages[2, :], num)
    Vz_sampled = resample(voltages[3, :], num)
    return np.array([t_sampled, Vx_sampled, Vy_sampled, Vz_sampled])

# Filter the voltages at a bandwidth
################################################################


def _butter_bandpass_filter(data, lowcut, highcut, fs):
    """subfunction of filt
    """
    b, a = butter(5, [lowcut / (0.5 * fs), highcut / (0.5 * fs)],
                  btype='band')  # (order, [low, high], btype)
    return lfilter(b, a, data)


def filters(voltages, FREQMIN=50.e6, FREQMAX=200.e6):
  """ Filter signal v(t) in given bandwidth 

  Parameters
  ----------
   : voltages
      The array of time (s) + voltage (muV) vectors to be filtered
   : FREQMIN 
      The minimal frequency of the bandpass filter (Hz)
   : FREQMAX: 
      The maximal frequency of the bandpass filter (Hz)
      
      
  Raises
  ------

  Notes
  -----
  At present Butterworth filter only is implemented

  Examples
  --------
  ```
  >>> from signal_treatment import _butter_bandpass_filter
  ```
  """

  t = voltages[:,0]
  v = np.array(voltages[:, 1:])  # Raw signal

  fs = 1 / np.mean(np.diff(t))  # Compute frequency step
  print("Trace sampling frequency: ",fs/1e6,"MHz")
  nCh = np.shape(v)[1]
  vout = np.zeros(shape=(len(t), nCh))
  res = t
  for i in range(nCh):
        vi = v[:, i]
        #vout[:, i] = _butter_bandpass_filter(vi, FREQMIN, FREQMAX, fs)
        res = np.append(res,_butter_bandpass_filter(vi, FREQMIN, FREQMAX, fs))
  
  res = np.reshape(res,(nCh+1,len(t)))  # Put it back inright format
  return res


##########################################################################
##########################################################################
