import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X=np.load("image.npz")["arr_0"]
y=pd.read_csv("labels.csv")["labels"]
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1,train_size=7500,test_size=2500)
X_train_scale=X_train/255.0
X_test_scale=X_test/255.0
clf=LogisticRegression().fit(X_train_scale,y_train)
def get_prediction(image):
    im_pil=Image.open(image)
    image_bw=im_pil.convert("L")
    image_bw_resize=image_bw.resize((28,28),Image.ANTIALIAS)
    pixel_filter=20
    minimum_pixels=np.percentile(image_bw_resize,pixel_filter)
    image_bw_resize_inverted_scale=np.clip(image_bw_resize-minimum_pixels,0,255)
    max_pixels=np.max(image_bw_resize)
    image_bw_resize_inverted_scale=np.asarray(image_bw_resize-minimum_pixels)/max_pixels
    test_sample=np.array(image_bw_resize_inverted_scale).reshape(1,784)
    test_pred=clf.predict(test_sample)
    return test_pred[0]