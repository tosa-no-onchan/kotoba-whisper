# -*- coding: utf-8 -*-
"""
kotoba-whisper/mic_stream.py

base from
/home/nishi/Documents/VisualStudio-TF/lstm_sound_to_text/inferencModel_mic.py

"""
import pyaudio
import numpy as np
import librosa
import threading

from time import sleep
from queue import Queue
import sys


class MicStream():
    def __init__(self,data_queue,rate=16000,level_th=600.0,level_stop_th=500.0,max_sec=3,low_sec=2) -> None:
        self.data_queue=data_queue
        self.rate=rate
        #self.chunk=int(rate/10)
        #self.chunk=int(rate/8)
        #self.chunk=int(rate/5)
        self.chunk=int(rate/4)
        #self.predict_dt = int(10*10)     # predict data length  -> 9 [sec]
        self.level_th=level_th
        self.level_stop_th=level_stop_th
        self.max_sec=max_sec

        self.max_len=int(rate*max_sec/self.chunk)
        self.low_len=int(rate*low_sec/self.chunk)
        if self.low_len ==0:
            self.low_len=1

    def callback(self,in_data, frame_count, time_info, status):
        # print(in_data)
        self.chunk_queue.put(in_data)

        return (in_data, pyaudio.paContinue)

    def start(self):
        self.chunk_queue = Queue()

        p=pyaudio.PyAudio()

        self.stream=p.open(format = pyaudio.paInt16,
                channels = 1,
                rate = self.rate,
                frames_per_buffer = self.chunk,
                input = True,
                output = False, # inputとoutputを同時にTrueにする
                stream_callback=self.callback
                )

        running = [True]
        def threaded_listen():
            s_on=False
            s_chk_cnt=0
            put_f=False
            dt=[]
            cnt_len=0
            while running[0]==True and self.stream.is_active():

                if not self.chunk_queue.empty():

                    #input = b''.join(self.chunk_queue.queue)
                    input=self.chunk_queue.get()
                    #y=data_queue.queue.pop()
                    #audio_data = b''.join(y)
                    #self.chunk_queue.queue.clear()


                    #input = self.stream.read(self.chunk)
                    dx = np.frombuffer(input, dtype='int16').astype('float16')

                    level_max = dx.max()
                    #print('level=',level_max)
                    if s_on==False:
                        if level_max > self.level_th:
                            dt.append(input)
                            s_on=True
                            s_chk_cnt=0
                            cnt_len=1
                            #print('level_max:',level_max)
                    else:
                        dt.append(input)
                        cnt_len+=1
                        # 会話中に一杯になった時、継続させないといけない。
                        if cnt_len >= self.max_len:
                            put_f=True
                            cnt_len=0
                            #s_on=False
                        else:
                            if level_max < self.level_stop_th:
                                #print('level_low:',level_max)
                                s_chk_cnt+=1
                                if s_chk_cnt >= self.low_len:
                                    s_on=False
                                    put_f=True
                            else:
                                s_chk_cnt=0
                            
                    if put_f==True:
                        #print('put len(dt):',len(dt))
                        data=b''.join(dt)
                        self.data_queue.put(data)
                        #self.data_queue.put(dt)
                        dt=[]
                        put_f=False

        def stopper(wait_for_stop=True):
            running[0] = False
            if wait_for_stop:
                listener_thread.join()  # block until the background thread is done, which can take around 1 second

        listener_thread = threading.Thread(target=threaded_listen)
        listener_thread.daemon = True
        listener_thread.start()
        return stopper



def speaker_test(test=False,np_data=None):

    #RATE=44100
    #RATE=22050
    RATE=16000

    if test==True:
        sample_mp3="/home/nishi/local/tmp/commonvoice/cv-corpus-11.0-2022-09-21/ja/common_voice_ja_32866812.mp3"
        print('load sample_mp3')
        dx, fs = librosa.load(sample_mp3)
        RATE=fs
        print('type(dx):',type(dx))
        # type(dx): <class 'numpy.ndarray'>
        print('dx.shape:',dx.shape)
        # dx.shape: (99225,)
        print('dx.dtype:',dx.dtype)
        # dx.dtype: float32
    else:
        dx=np_data
        print('type(dx):',type(dx))
        # type(dx): <class 'numpy.ndarray'>
        print('dx.shape:',dx.shape)
        # dx.shape: (52800,)
        print('dx.dtype:',dx.dtype)
        # dx.dtype: float32

    #CHUNK=1024
    CHUNK=2**11
    ps=pyaudio.PyAudio()

    stream=ps.open(format = pyaudio.paInt16,
            channels = 1,
            rate = RATE,
            frames_per_buffer = CHUNK,
            input = False,
            output = True) # inputとoutputを同時にTrueにする

    #audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    #dx=audio_np
    j=dx*2**15
    j=j.astype('int16')
    #print(j.shape)
    output = stream.write(j,j.shape[0])


if __name__ == "__main__":

    if False:
        speaker_test(test=True)
        sys.exit()

    data_queue = Queue()

    mic_stream=MicStream(data_queue)
    stopper = mic_stream.start()

    while True:
        try:
            if not data_queue.empty():
                #audio_data = b''.join(data_queue.queue)
                #data_queue.queue.clear()

                print('get queue')
                audio_data = data_queue.get()

                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                speaker_test(test=False,np_data=audio_np)


        except KeyboardInterrupt:
            stopper()
            break

        sleep(2.0)
