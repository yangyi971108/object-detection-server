import requests
import argparse
import cv2
import numpy as np
import os

PyTorch_REST_API_URL = "http://127.0.0.1:5000/predict"

def vis_detections1(im,dets):
    cnt = 0
    for i in range(np.minimum(500,dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i,:4])
        cnt += 1
        cv2.circle(im,(int((bbox[2]+bbox[0])/2), int((bbox[3]+bbox[1])/2)), 12, (0, 0, 255), -1)
    cv2.putText(im, str(cnt), (40, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 204, 0), 2)
    return im


def predict_result():

    r = requests.get("http://10.181.6.133:5000/predict/Linked_List_List_44.png").json();
    print(r)
    det_box = []
    for (i, result) in enumerate(r['predictions']):
        box = []
        if i > 0:
            [box.append(int(j)) for j in result["BoxList"]]
        det_box.append(box)
    del[det_box[0]]
    print("det_box",det_box)
    print(cv2.imread('Linked_List_List_44.png'))
    im2show = vis_detections1(cv2.imread('Linked_List_List_44.png'), np.array(det_box))
    result_path = os.path.join('image_det' +"Linked_List_List_44.jpg")
    cv2.imwrite(result_path, im2show)
    cv2.imshow('imshow', im2show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

predict_result()
