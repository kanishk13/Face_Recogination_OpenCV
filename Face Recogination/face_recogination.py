#Recognise Face using Classification Algorithm -KNN 

# 1.Read A Video Stream using OpenCV
# 2.Extract faces out of it
# 3.Load Training data (numpy array of all the persons )
	# x-values are stored in the numpy arrays 
	# y-values we need to assign for each person 
#4. use knn to find the prediction to face (int)
#5.map to predicit id to the name of the user 
#6. Display the predicitions on the screen - bounding box and name 

import cv2
import numpy as np
import os 


def distance(v1,v2):
	return np.sqrt(((v1-v2)**2).sum())
		
def knn(train,test,k=5):
	dist=[]
	
	for i in range(train.shape[0]):
		ix = train[i, :-1]
		iy =train[i, -1]
		d =distance(test,ix)
		dist.append([d,iy])
	
	dk=sorted(dist,key=lambda x:x[0])[:k]
	labels =np.array(dk)[:,-1]
	output=np.unique(labels,return_counts=True)
	index=np.argmax(output[1])
	
	return output[0][index]



cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)


face_cascade = cv2.CascadeClassifier("C:\\Users\\kanishk vohra\\Desktop\\Face Recogination\\haarcascade_frontalface_alt.xml")

skip = 0
dataset_path ='C:\\Users\\kanishk vohra\\Desktop\\Face Recogination\\data\\'  

face_data = []
labels=[]

class_id = 0 
names={} 

for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		names[class_id]=fx[:-4]
		print("loaded "+fx)
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)
		target = class_id*np.ones((data_item.shape[0],))
		class_id+=1
		labels.append(target)
	
face_dataset = np.concatenate(face_data,axis=0)
face_lables=np.concatenate(labels,axis=0).reshape((-1,1))	

print(face_dataset.shape)
print(face_lables.shape)

trainset = np.concatenate((face_dataset,face_lables),axis=1)
print(trainset.shape)


while True:
	ret,frame= cap.read()
	if ret== False:
		continue
	
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	
	for face in faces:
		x,y,w,h =face
		offset = 10 
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		out=knn(trainset,face_section.flatten())
		pred_name=names[int(out)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
	
	cv2.imshow("Frames",frame)
	
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
	    break
		
cap.release()
cv2.destroyAllWindows()		
