#Recognise Face using Classification Algorithm -KNN 

# 1.Load Training data (numpy array of all the persons )
	# x-values are stored in the numpy arrays 
	# y-values we need to assign for each person 

# 2.Read A Video Stream using OpenCV
# 3.Extract faces out of it
# 4. use knn to find the prediction to face (int)
# 5.map to predicit id to the name of the user 
# 6. Display the predicitions on the screen - bounding box and name 

import cv2
import numpy as np
import os 


def distance(v1,v2):                    #takes two vector vector v1 ma
	return np.sqrt(((v1-v2)**2).sum())  #Euclidean distance formula 
		
def knn(train,test,k=5):   		 # 5 query points 
	dist=[]                      # distance array  
	
	for i in range(train.shape[0]):       
		ix = train[i, :-1]            # point in the dataset 
		iy =train[i, -1]
		d =distance(test,ix)        
		dist.append([d,iy])        
	
	dk=sorted(dist,key=lambda x:x[0])[:k]     # sort according the distance 
	labels =np.array(dk)[:,-1]
	output=np.unique(labels,return_counts=True)
	index=np.argmax(output[1])             
	
	return output[0][index]



cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

#to extract the faces 
face_cascade = cv2.CascadeClassifier("C:\\Users\\kanishk vohra\\Desktop\\Face Recogination\\haarcascade_frontalface_alt.xml")

skip = 0
dataset_path ='C:\\Users\\kanishk vohra\\Desktop\\Face Recogination\\data\\'  

face_data = []          # training data load all the files x of the data 
labels=[]				# y values of our data 

class_id = 0            # file one will be intlized with zero  lables for tha given file 
names={} 

#data prepration 

for fx in os.listdir(dataset_path):    # all the files in folder path 
	if fx.endswith('.npy'):            # if your file ends with numpy file 
		names[class_id]=fx[:-4]        # remove the last four character to get the name of the file ex kanishk.npy        
		print("loaded "+fx)
		data_item = np.load(dataset_path+fx)  	# file name along with the name 
		face_data.append(data_item)			 	# faces list
		target = class_id*np.ones((data_item.shape[0],))    #
		class_id+=1    							# incrementing the class id by one 
		labels.append(target)					#computing the lablesa and saving it     
												
face_dataset = np.concatenate(face_data,axis=0)
face_lables=np.concatenate(labels,axis=0).reshape((-1,1))	

print(face_dataset.shape)      #(faces,features)
print(face_lables.shape) 		

trainset = np.concatenate((face_dataset,face_lables),axis=1)  	
print(trainset.shape)   										


while True:
	ret,f rame= cap.read()
	if ret== False:      # if we can't detect the face  then recapture 
		continue
						# if we detected the face 								
	faces = face_cascade.detectMultiScale(frame,1.3,5)   # giving the frame scaling 
	
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
