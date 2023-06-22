
import sys
import socket
import struct
import time

HOST = 'localhost'
PORT = 2000

timer = 4 #Minutes
samp_rate = 1024 #samples per second


with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.bind((HOST, PORT))
    print(f'Binding socket to... {HOST} : {PORT}, Getting 1024 samples every second for {timer} minutes')

    startTime = currTime = time.time()
    with open("Samples(Splotches)", 'wb') as bin_file:
        while time.time() - startTime <= 60*timer:
            data = s.recv(8192)
            if(time.time() - currTime > 1):
                bin_file.write(data)
                currTime = time.time()
                print('Saving data..')
            

            #s.flush()
            #bin_file.write(data)
        

        
