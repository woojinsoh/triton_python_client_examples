import numpy as np
import soundfile

class AudioSegment(object):
    """
    Monaural audio segment abstraction.
    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :type sample_rate: int
    :raises TypeError: If the sample data type is not float or int.
    """

    def __init__(self, samples, sample_rate, target_sr=16000, trim=False,
                 trim_db=60):
        """Create audio segment from samples.
        Samples are convert float32 internally, with int scaled to [-1, 1].
        """
        samples = self._convert_samples_to_float32(samples)
        self._samples = samples
        self._sample_rate = sample_rate
        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)

    @staticmethod
    def _convert_samples_to_float32(samples):
        """
        Convert sample type to float32.
        Audio sample type is usually integer or float-point.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype('float32')
        if samples.dtype in np.sctypes['int']:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= (1. / 2 ** (bits - 1))
        elif samples.dtype in np.sctypes['float']:
            pass
        else:
            raise TypeError("Unsupported sample type: %s." % samples.dtype)
        return float32_samples

    @classmethod
    def from_file(cls, filename, target_sr=16000, int_values=False, offset=0,
                  duration=0, trim=False):
        """
        Load a file supported by librosa and return as an AudioSegment.
        :param filename: path of file to load
        :param target_sr: the desired sample rate
        :param int_values: if true, load samples as 32-bit integers
        :param offset: offset in seconds when loading audio
        :param duration: duration in seconds when loading audio
        :return: numpy array of samples
        """
        with soundfile.SoundFile(filename, 'r') as f:
            dtype = 'int32' if int_values else 'float32'
            sample_rate = f.samplerate
            if offset > 0:
                f.seek(int(offset * sample_rate))
            if duration > 0:
                samples = f.read(int(duration * sample_rate), dtype=dtype)
            else:
                samples = f.read(dtype=dtype)

        samples = samples.transpose()
        return cls(samples, sample_rate, target_sr=target_sr, trim=trim)

    @property
    def samples(self):
        return self._samples.copy()

    @property
    def sample_rate(self):
        return self._sample_rate


def ctc_decoder_predictions_tensor(prediction_cpu_tensor, labels):
    """
    Takes output of greedy ctc decoder and performs ctc decoding algorithm to
    remove duplicates and special symbol. Returns prediction
    Args:
        tensor: model output tensor
        label: A list of labels
    Returns:
        prediction
    """
    blank_id = len(labels) - 1
    hypotheses = []
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    prediction = prediction_cpu_tensor.tolist()

    # CTC decoding procedure
    decoded_prediction = []
    previous = len(labels) - 1 # id of a blank symbol
    for p in prediction:
        if (p != previous or previous == blank_id) and p != blank_id:
            decoded_prediction.append(p)
        previous = p
    hypothesis = ''.join([labels_map[c] for c in decoded_prediction])
    hypotheses.append(hypothesis)    

    return hypotheses


def get_audio_chunk_from_soundfile(sf, chunk_size):
    """
    Read audio chunk from a file to make streaming input.
    Args:
        sf: original audio from soundfile
        chunk_size: size of the chunk
    Return:
        audio_signal: chunked audio
        end: flag to check if it's the last chunk
    """

    dtype_options = {'PCM_16': 'int16', 'PCM_32': 'int32', 'FLOAT': 'float32'}
    dtype_file = sf.subtype
    if dtype_file in dtype_options:
        dtype = dtype_options[dtype_file]
    else:
        dtype = 'float32'
    audio_signal = sf.read(chunk_size, dtype=dtype)
    end = False

    # pad to chunk size
    if len(audio_signal) < chunk_size:
        end = True
        audio_signal = np.pad(audio_signal, (0, chunk_size-len(
            audio_signal)), mode='constant')
    return audio_signal, end


# generator that returns chunks of audio data from file
def audio_generator_from_file(input_filename, target_sr, chunk_duration):
    """
    Generator that returns chunk of audio from file
    Args:
        input_filename: filename
        target_sr: target sample rate
        chunk_duration: size of chunk in seconds
    Return:
        audio_segment.samples: chunked audio sample
        target_sr: target sample rate
        start: Flag to check if it's the first chunk
        end: Flag to check if it's the last chunk
    """

    sf = soundfile.SoundFile(input_filename, 'rb')
    chunk_size = int(chunk_duration*sf.samplerate)
    start = True
    end = False

    while not end:

        audio_signal, end = get_audio_chunk_from_soundfile(sf, chunk_size)

        audio_segment = AudioSegment(audio_signal, sf.samplerate, target_sr)

        yield audio_segment.samples, target_sr, start, end
        start = False

    sf.close()


def audio_generator_from_file_with_buffer(input_filename, target_sr, chunk_duration, n_chunk_overlap):
    """
    Generator that returns chunk of audio with buffer from file 
    Args:
        input_filename: filename
        target_sr: target sample rate
        chunk_duration: size of chunk in seconds
        n_chunk_overlap: 
    Return:
        audio_segment.samples: chunked audio sample
        target_sr: target sample rate
        start: Flag to check if it's the first chunk
        end: Flag to check if it's the last chunk
    """

    sf = soundfile.SoundFile(input_filename, 'rb')
    chunk_size = int(chunk_duration*sf.samplerate)
    start = True
    end = False

    audio_signal_with_buffer = np.zeros(shape=chunk_size + 2 * n_chunk_overlap, dtype=np.float32)

    while not end:

        audio_signal_with_buffer[-chunk_size:], end = get_audio_chunk_from_soundfile(sf, chunk_size)

        audio_segment = AudioSegment(audio_signal_with_buffer, sf.samplerate, target_sr)
        #audio_features, features_length = get_speech_features(audio_segment.samples, target_sr, speech_features_params)
        #yield audio_features, start, end
        yield audio_segment.samples, target_sr, start, end
        start = False
        audio_signal_with_buffer[:-chunk_size] = audio_signal_with_buffer[chunk_size:]

    sf.close()


def postprocess(transcript_values, labels):
    """
    Decoding predicted values to the queires
    """
    res = []
    for transcript in transcript_values:
        t=ctc_decoder_predictions_tensor(transcript, labels)
        res.append(t)
    return res