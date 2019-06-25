import sys
import os
import glob
import subprocess
import re

def get_lock_image(i):

    #bscript = '#!/bin/bash\n./darknet detect /home/ubuntu/darknet/AlexeyAB/darknet/build/darknet/x64/cfg/yolo-obj.cfg' + ' ' + \
    #'/home/ubuntu/darknet/AlexeyAB/darknet/build/darknet/x64/backup/yolo-obj_final.weights' + ' ' + '"' + i + '"'
    result = subprocess.run(['./darknet',
                    'detect',
                    '/home/ubuntu/darknet/AlexeyAB/darknet/build/darknet/x64/cfg/yolo-obj.cfg',
                    '/home/ubuntu/darknet/AlexeyAB/darknet/build/darknet/x64/backup/yolo-obj_final.weights',
                    i], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    print(output)
    m = re.findall('lock:', output)  
    print('we found ' + str(len(m)) + ' lock(s)')
    if len(m) == 1:
       racks = re.findall('rack:', output)
       print('and ' + str(len(racks)) + ' rack(s)')
       lines = len(m) + len(racks)
       for line in output.splitlines():
           if re.findall('lock:', line):
                print('yes')
                print(line)
           else:
               print('no') 

       return {'racks': len(racks)}
    return False

def get_cropped_image():
    lst = []
    list_of_files = os.listdir("result_img")
    for i in list_of_files:
        lst = lst + re.findall(r'img_\d_\d_\d_lock.*', i)
    return lst

get_lock_image('/home/ubuntu/darknet/test/IMG_2277.jpg')
   # cropped_images = get_cropped_image()
   # print(cropped_images)
