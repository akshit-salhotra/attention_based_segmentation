import cv2
video_path='to_akshit/case16_10sec_ir.mp4'
cap=cv2.VideoCapture(video_path)
count=0
while cap.isOpened():
    ret,frame=cap.read()
    # if count>=8700 and count<=10050:
    #     if count%3==0:
        # print(ret)
    cv2.imwrite(f'ir/{count}.jpg',frame)
    count+=1
    
    