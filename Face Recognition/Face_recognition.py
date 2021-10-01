import cv2
import numpy as np
import os

print("Press 'q' to stop recording")

############# KNN CODE ##############
def distance(v1,v2):
    #Eucledian
    return np.sqrt(((v1-v2)**2).sum())
def knn(train,test,k=5):
    dist = []
    for i in range(train.shape[0]):
        #Get the vector and label
        ix = train[i,:-1]
        iy = train[i,-1]
        #Compute the distance from point
        d=distance(test,ix)
        dist.append([d,iy])
    #sort based on distance and get top k
    dk = sorted(dist,key=lambda x: x[0])[:k]
    #Retrive only one labels
    labels = np.array(dk)[:,-1]
    #Get frequencies of each label
    output = np.unique(labels,return_counts=True)
    #Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]
############################################################

#Init Camera
cap = cv2.VideoCapture(0)

#Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip =0
dataset_path = './data/'
face_data =[]
labels=[]
class_id=0  #Labels for the given file
names={}  #Map[ping btw id - name]

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        #Creat a mapping btw class_id and name
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        #Create LAbels for the class

        target = class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

# print(face_dataset.shape)
# print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

#Testing 
while True:
    ret,frame =cap.read()
    if ret==False:
        continue
    
    # gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    
    for face in faces[-1:]:
        x,y,w,h=face
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),1)

        offset = 10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        #Predicted Label(out)
        out = knn(trainset,face_section.flatten())

        #Display on the screen the name and rectangle around it

        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x+5,y-10),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,200,0),1,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),1)
    cv2.imshow("Faces",frame)
    key = cv2.waitKey(1) & 0xFF
    if(key==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()