import pyaudio
import threading
import time
import argparse
import wave
import torchaudio
import torch
import sys
import numpy as np
from neuralnet.dataset import get_featurizer
from decoder import DecodeGreedy, CTCBeamDecoder
from threading import Event


class Listener:
    """
    Class to handle audio input streaming using PyAudio.
    Attributes:
        listener (Listener): Instance of the Listener class for audio input streaming.
        model (torch.jit.ScriptModule): Pre-trained model for speech recognition.
        featurizer (function): Function to extract features from audio waveforms.
        audio_q (list): List to store audio input data.
        hidden (tuple): Tuple containing hidden states for the model.
        beam_results (str): Result of the beam search decoding.
        out_args (torch.Tensor): Output tensor from the model.
        beam_search (CTCBeamDecoder): Instance of the CTCBeamDecoder class for beam search decoding.
        context_length (int): Length of the context window in frames.
        start (bool): Flag to indicate if the speech recognition engine has started.
    """

    def __init__(self, sample_rate=8000, record_seconds=2):
        """
        Initializes the Listener object with specified parameters.

        Args:
            sample_rate (int): Sampling rate for audio input (default: 8000 Hz).
            record_seconds (int): Duration of each audio recording in seconds (default: 2 seconds).
        """
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.sample_rate,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.chunk)

    def listen(self, queue):
        """
        Continuously listens for audio input and appends it to the provided queue.

        Args:
            queue (list): List to store audio input data.
        """
        while True:
            data = self.stream.read(self.chunk , exception_on_overflow=False)
            queue.append(data)
            time.sleep(0.01)

    def run(self, queue):
        """
        Starts a new thread to listen for audio input.

        Args:
            queue (list): List to store audio input data.
        """
        thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
        thread.start()
        print("\nSpeech Recognition engine is now listening... \n")


class SpeechRecognitionEngine:
    """
    Class to perform speech recognition using a pre-trained model.
    """
    def __init__(self, model_file, ken_lm_file, context_length=10):
        """
        Initializes the SpeechRecognitionEngine with the specified parameters.

        Args:
            model_file (str): Path to the optimized model file.
            ken_lm_file (str): Path to the KenLM language model file.
            context_length (int): Length of the context window in seconds (default: 10 seconds).
        """
        self.listener = Listener(sample_rate=8000)
        self.model = torch.jit.load(model_file)
        self.model.eval().to('cpu')  # Run on cpu
        self.featurizer = get_featurizer(8000)
        self.audio_q = list()
        self.hidden = (torch.zeros(1, 1, 1024), torch.zeros(1, 1, 1024))
        self.beam_results = ""
        self.out_args = None
        self.beam_search = CTCBeamDecoder(beam_size=100, kenlm_path=ken_lm_file)
        self.context_length = context_length * 50 # multiply by 50 because each 50 from output frame is 1 second
        self.start = False

    def save(self, waveforms, fname="audio_temp"):
        """
        Saves audio waveforms to a WAV file.

        Args:
            waveforms (list): List of audio waveforms.
            fname (str): File name for the saved WAV file (default: "audio_temp").

        Returns:
            str: File path of the saved WAV file.
        """
        wf = wave.open(fname, "wb")
        # set the channels
        wf.setnchannels(1)
        # set the sample format
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        # set the sample rate
        wf.setframerate(8000)
        # write the frames as bytes
        wf.writeframes(b"".join(waveforms))
        # close the file
        wf.close()
        return fname

    def predict(self, audio):
        """
        Performs speech recognition on the provided audio data.

        Args:
            audio (list): List of audio waveforms.

        Returns:
            tuple: A tuple containing the recognized text and the current context length in seconds.
        """
        with torch.inference_mode():
            fname = self.save(audio)
            waveform, _ = torchaudio.load(fname)  # don't normalize on train
            log_mel = self.featurizer(waveform).unsqueeze(1)
            out, self.hidden = self.model(log_mel, self.hidden)
            out = torch.nn.functional.softmax(out, dim=2)
            out = out.transpose(0, 1)
            self.out_args = out if self.out_args is None else torch.cat((self.out_args, out), dim=1)
            results = self.beam_search(self.out_args)
            current_context_length = self.out_args.shape[1] / 50  # in seconds
            if self.out_args.shape[1] > self.context_length:
                self.out_args = None
            return results, current_context_length

    def inference_loop(self, action):
        """
        Continuously performs speech recognition on the audio queue.

        Args:
            action (function): Function to perform after speech recognition.
        """
        while True:
            if len(self.audio_q) < 5:
                continue
            else:
                pred_q = self.audio_q.copy()
                self.audio_q.clear()
                action(self.predict(pred_q))
            time.sleep(0.05)

    def run(self, action):
        """
        Starts the speech recognition engine.

        Args:
            action (function): Function to perform after speech recognition.
        """
        self.listener.run(self.audio_q)
        thread = threading.Thread(target=self.inference_loop,
                                    args=(action,), daemon=True)
        thread.start()


class DemoAction:
    """
    Class to handle the action after performing speech recognition.
    """
    def __init__(self):
        self.asr_results = ""
        self.current_beam = ""

    def __call__(self, x):
        """
        Combines the current and previous results and prints them.

        Args:
            x (tuple): A tuple containing the recognized text and the current context length.

        Returns:
            None
        """
        # Unpack the tuple
        results, current_context_length = x

        # Combine the current beam results with previous results and print transcript
        self.current_beam = results
        transcript = " ".join(self.asr_results.split() + results.split())
        print(transcript)

        # Update the current transcript
        if current_context_length > 10:
            self.asr_results = transcript

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demoing the speech recognition engine in terminal.")
    parser.add_argument('--model_file', type=str, default=None, required=True,
                        help='optimized file to load. use freeze_model.py')
    parser.add_argument('--ken_lm_file', type=str, default=None, required=False,
                        help='If you have an ngram lm use to decode')

    args = parser.parse_args()

    # activate speech recognition engine
    asr_engine = SpeechRecognitionEngine(args.model_file, args.ken_lm_file)
    action = DemoAction()

    # Start the speech recognition engine 
    # and wait for the threading event to keep the program running
    asr_engine.run(action)
    threading.Event().wait()
