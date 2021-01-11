import socket
import time

TCP_IP = '192.168.125.1'
TCP_PORT = 1025
BUFFER_SIZE = 1024


# ______________Input the object locating algorithm here_________________#

def getobjectpos():
    object_pos = [600, 350, 250]  # Replace object position here
    sent_data = str(object_pos)
    time.sleep(1)
    return sent_data


# __________________________00 THE END 00________________________________#

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))
while 1:
    receive = s.recv(BUFFER_SIZE).decode('utf-8')
    if (receive == "0xFD"):
        print("Object delivered/n")
    elif (receive == "0xFC"):
        print("Stand by/n")
    elif (receive == "0xFF"):
        data = getobjectpos()
        print("Object location request/n")
        s.send(data.encode())
    elif (receive == "0xFB"):
        print("Moving../n")
    elif (receive == "0xFA"):
        print("server closed/n")
        break;
    elif (receive == ""):
        pass;
    else:
        print(".")
s.close()
