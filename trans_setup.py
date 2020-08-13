#1920, 1080

import os

# print (len(os.walk('/home/deepmeta/data/images').__next__()[2]))
# filecount = len(os.walk('/home/deepmeta/data/images').__next__()[2]) / 2
# filecount = int(filecount)
width = '1920'
height = '1080'
fr_width = 1920
fr_height = 1080
#print(filecount)

path = "/home/deepmeta/detectron2/baseball_2019_setup_without_0/"
file_list_txt = []
file_list_jpg = []
jsonString = '{"images":['
i = 0
y=0
z=0
annotations = '"annotations":['

for root, dirs, files in os.walk(path):

    for file in files:
#         print("root : " + str(root))
#         print("dirs : " + str(dirs))
#         print("file : " + str(file))
        s = os.path.splitext(file)
#         print("s : " + str(s))
        
        
        if s[1] == '.jpg':
            file_list_jpg.append(root + "/" + file)
            jsonString = jsonString + '{"height":' + height + ',"width":' + width + ',"id":' + str(i) + ',"file_name":"' + root + "/" + file + '"},'
            i = i + 1                 
            
        if s[1] == '.txt':
            file_list_txt.append(root + "/" + file)
            print("test : " + root + "/" + file)
            f = open(root + "/" + file, 'r')
            z = z + 1
            while True:

                line = f.readline()
                if not line or line == "\n":
                    break
#                 print("line : " + str(line))

                annotations = annotations + '{"image_id":' + str(z) + ',"bbox":['
                tmplst = line.split(' ')
                classid = int(tmplst[0])
#                 print('classid : ' + str(classid))
                x_yolo = float(tmplst[1])
                y_yolo = float(tmplst[2])
                w_yolo = float(tmplst[3])
                h_yolo = float(tmplst[4])        

                x1 = (int)((x_yolo - w_yolo/2) * fr_width)
                y1 = (int)((y_yolo - h_yolo/2) * fr_height)
                x2 = (int)((x_yolo + w_yolo/2) * fr_width)
                y2 = (int)((y_yolo + h_yolo/2) * fr_height)       

                x1 = x1
                y1 = y1
                x2 = x2 - x1
                y2 = y2 - y1

#                 print(x1)
#                 print(y1)
#                 print(x2)
#                 print(y2)
                #classnote = '{}({})'.format(classes[classid], classid)

                annotations = annotations + str(x1) + '.0,' + str(y1) + '.0,' + str(x2) + '.0,' + str(y2) + '.0],"category_id":' + str(classid+1) + ',"id":' + str(y+1) + '},'
                y = y + 1

            f.close()
         
            
   
        
                  
jsonString = jsonString[:-1]
jsonString = jsonString + '],'  

#classes = ['scoreboard', 'homerun', 'setup', 'pitchingzone', 'interview', 'pitcherwindup', 'pitchersetup', 'divingcatch', 'video', 'MBCSPORTSstart', 'KBSNSPORTSstart', 'SBSSPORTSstart', 'SPOTVstart', 'SPOTV2start', 'MBCstart', 'KBSstart', 'SBSstart']
classes = ['pitchersetup', 'pitcherwindup', 'setup']
    
categories = '"categories":['
print(categories)
print (len(classes))

for j in range(len(classes)):
    if j+1 == len(classes):
        categories = categories + '{"supercategory":"' + classes[j] + '","id":' + str(j+1) + ',"name":"' + classes[j] + '"}],'
    else:
        categories = categories + '{"supercategory":"' + classes[j] + '","id":' + str(j+1) + ',"name":"' + classes[j] + '"},'

jsonString = jsonString + categories


annotations = annotations[:-1]
annotations = annotations + ']}'

print(annotations)
jsonString = jsonString + annotations
print('--------------------------------------------------------')
print(jsonString)

text = open('/home/deepmeta/detectron2/baseball_2019_setup_without_0/result.json', 'w')
text.write(jsonString)
text.close()
# cv2.putText(img, classnote, (x1+5, y1+20), cv2.FONT_HERSHEY_COMPLEX, 0.75, yellow, 2)

