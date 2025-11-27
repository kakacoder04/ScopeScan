from keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("model.h5")
img = image.load_img('./input/test/2_polyps/test_polyps_ (8).jpg',target_size=(224,224))
imag = image.img_to_array(img)
imaga = np.expand_dims(imag,axis=0) 
ypred = model.predict(imaga)

print(ypred)

a=np.argmax(ypred,-1)

if a==0:
    op="Normal"
elif a==1:
    op="Ulcerative Colitis"
elif a==2:
    op="Polyp"
else:
    op="Esophagitis"  

plt.imshow(img)
print("THE UPLOADED IMAGE SEEMS TO BE: "+str(op))  