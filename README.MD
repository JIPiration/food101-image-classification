

```python
# Check if GPU is enabled
import tensorflow as tf
print(tf.__version__)
print(tf.test.gpu_device_name())
```

    1.13.1

# Food101 Dataset을 이용한 Classification
###DoingLAB_

Dataset 중 알파벳 순으로 상위 50개 클래스 사용하기



```python
import os
os.listdir('food-101/images')[:]
```
    ['apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare',
     'beet_salad','beignets','bibimbap','bread_pudding','breakfast_burrito',
     'bruschetta','caesar_salad','cannoli','caprese_salad','carrot_cake',
     'ceviche','cheesecake','cheese_plate','chicken_curry','chicken_quesadilla',
     'chicken_wings','chocolate_cake','chocolate_mousse','churros','clam_chowder',
     'club_sandwich','crab_cakes','creme_brulee','croque_madame','cup_cakes',
     'deviled_eggs','donuts','dumplings','edamame','eggs_benedict',
     'escargots','falafel','filet_mignon','fish_and_chips','foie_gras',
     'french_fries','french_onion_soup','french_toast','fried_calamari','fried_rice',
     'frozen_yogurt','garlic_bread','gnocchi','greek_salad','grilled_cheese_sandwich',
     'grilled_salmon','guacamole','gyoza','hamburger','hot_and_sour_soup',
     'hot_dog','huevos_rancheros','hummus','ice_cream','lasagna','lobster_bisque',
     'lobster_roll_sandwich','macaroni_and_cheese','macarons','miso_soup','mussels',
     'nachos','omelette','onion_rings','oysters','pad_thai',
     'paella','pancakes','panna_cotta','peking_duck','pho',
     'pizza','pork_chop','poutine','prime_rib','pulled_pork_sandwich',
     'ramen','ravioli','red_velvet_cake','risotto','samosa',
     'sashimi','scallops','seaweed_salad','shrimp_and_grits','spaghetti_bolognese',
     'spaghetti_carbonara','spring_rolls','steak','strawberry_shortcake','sushi',
     'tacos','takoyaki','tiramisu','tuna_tartare','waffles']

```python
os.listdir('food-101/meta')
```


    ['classes.txt',
     'labels.txt',
     'test.json',
     'test.txt',
     'train.json',
     'train.txt']

**
상위 50개 푸드 클래스 랜덤 이미지 시각화**

```python
import matplotlib.pyplot as plt
import matplotlib.image as img
%matplotlib inline
import numpy as np
from collections import defaultdict
import collections
import os
```

```python
# 상위 50개의 클래스만 사용할 때, 
rows = 10
cols = 5
fig, ax = plt.subplots(rows, cols, figsize=(25,25))
fig.suptitle("Showing one random image from each class", y=1.05, fontsize=24) # Adding  y=1.05, fontsize=24 helped me fix the suptitle overlapping with axes issue
data_dir = "food-101/images/"
foods_sorted = sorted(os.listdir(data_dir))

foods_sorted = foods_sorted[:50]

food_id = 0
for i in range(rows):
  for j in range(cols):
    try:
      food_selected = foods_sorted[food_id] 
      food_id += 1
    except:
      break
    food_selected_images = os.listdir(os.path.join(data_dir,food_selected)) 
    food_selected_random = np.random.choice(food_selected_images) 

    img = plt.imread(os.path.join(data_dir,food_selected, food_selected_random))
    ax[i][j].imshow(img)
    ax[i][j].set_title(food_selected, pad = 10)
    
plt.setp(ax, xticks=[],yticks=[])
plt.tight_layout()

```


![50개 푸드 클래스 랜덤 이미지](https://user-images.githubusercontent.com/50260643/62836460-226e0100-bc9e-11e9-8be3-c80c174897cd.png)



```python

# 사용할 푸드 클래스만큼의 train, test 데이터셋 폴더를 위한
# 지정한 number수 만큼 생성
from shutil import copy
def prepare_data(filepath, src, dest, number):
  classes_images = defaultdict(list)
  with open(filepath, 'r') as txt:
      paths = [read.strip() for read in txt.readlines()]
      for p in paths:
        food = p.split('/')
        classes_images[food[0]].append(food[1] + '.jpg')
        
  count = 0
  for food in classes_images.keys():
    print("\nCopying images into ",food)
    if not os.path.exists(os.path.join(dest,food)):
      os.makedirs(os.path.join(dest,food))
    for i in classes_images[food]:
      copy(os.path.join(src,food,i), os.path.join(dest,food,i))
    
    count += 1
    if count == number:
        break
  print("Copying 끝!")
```


```python
# food-101/images 중 train할 파일 데이터 셋 준비
print("Creating train data...")
prepare_data('food-101/meta/train.txt', 'food-101/images', 'food-101/train', 50)
```

    Creating train data...
    Copying images into  apple_pie
    Copying images into  baby_back_ribs
    Copying images into  baklava
    Copying images into  beef_carpaccio
    Copying images into  beef_tartare
    Copying images into  beet_salad
    Copying images into  beignets
    .
    .
    .
    .
    .
    Copying images into  frozen_yogurt
    Copying images into  garlic_bread
    Copying images into  gnocchi
    Copying images into  greek_salad
    Copying images into  grilled_cheese_sandwich
    
    Copying 끝!

```python
# food-101/images 중 test할 파일 데이터 셋 준비
print("Creating test data...")
prepare_data('food-101/meta/test.txt', 'food-101/images', 'food-101/test', 50)
```

    Creating test data...
    Copying images into  apple_pie
    Copying images into  baby_back_ribs
    Copying images into  baklava
    Copying images into  beef_carpaccio
    Copying images into  beef_tartare
    Copying images into  beet_salad
    Copying images into  beignets
    .
    .
    .
    .
    .
    .
    .
    Copying images into  frozen_yogurt
    Copying images into  garlic_bread
    Copying images into  gnocchi
    Copying images into  greek_salad
    Copying images into  grilled_cheese_sandwich
    Copying 끝!



```python
# List of all 50 types of foods(sorted alphabetically)
print(foods_sorted)
print("len(foods_sorted): ", len(foods_sorted))
```

    ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich']
    len(foods_sorted):  50
    


```python
# 데이터셋 활용을 위한 data mini samples(train, test) 만들기
from shutil import copytree, rmtree
def dataset_mini(food_list, src, dest):
  if os.path.exists(dest):
    rmtree(dest) # removing dataset_mini(if it already exists) folders so that we will have only the classes that we want
  os.makedirs(dest)
  for food_item in food_list :
    print("Copying images into",food_item)
    copytree(os.path.join(src,food_item), os.path.join(dest,food_item))
```


```python
# mini file 생성을 위한, 먼저 3개 samples 생성.
food_list = ['bibimbap', 'eggs_benedict', 'grilled_cheese_sandwich']
src_train = 'food-101/train'
dest_train = 'food-101/train_mini'
src_test = 'food-101/test'
dest_test = 'food-101/test_mini'
```


```python
print("Creating train data folder with new classes")
dataset_mini(food_list, src_train, dest_train)
```

    Creating train data folder with new classes
    Copying images into bibimbap
    Copying images into eggs_benedict
    Copying images into grilled_cheese_sandwich
    


```python
print("Creating test data folder with new classes")
dataset_mini(food_list, src_test, dest_test)
```

    Creating test data folder with new classes
    Copying images into bibimbap
    Copying images into eggs_benedict
    Copying images into grilled_cheese_sandwich
    



```python
# 갯수 파악 'food-101/train', 'food-101/test'
train_file_count = sum(len(files) for _, _, files in os.walk(r'food-101/train_mini'))  # 'food-101/train'
print("train_file_count: ", train_file_count)

test_file_count = sum(len(files) for _, _, files in os.walk(r'food-101/test_mini'))  # 'food-101/test'
print("test_file_count: ", test_file_count)
```

    train_file_count:  2250
    test_file_count:  750
    


```python
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow import keras
import numpy as np
```


```python
# Inceptionv3 model을 활용한 3개의 푸드 클래스 pretrained
n = 3
K.clear_session()

import time
startTime = time.time()

n_classes = n  # 푸드 클래스에 따른 변수
img_width, img_height = 299, 299
train_data_dir = 'food-101/train_mini'
validation_data_dir = 'food-101/test_mini'
nb_train_samples = train_file_count  #test:2250, 50개:37500, total:75750  per:750
nb_validation_samples = test_file_count  #test:750, 50개:12500, total:25250  per:250
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

inception = InceptionV3(weights='imagenet', include_top=False)
x = inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(n ,kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)  # n = 푸드 클래스 갯수에 따른

model = Model(inputs=inception.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='best_model_3class.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history_3class.log')

print("epochs=3으로 먼저 테스트")
history = model.fit_generator(train_generator,
                    steps_per_epoch = nb_train_samples // batch_size,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    epochs=3, # 1 epoch 당 약 3시간 소요
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])
model.save('model_trained_3class.hdf5')
endTime = time.time() - startTime
print(endTime)
```

    Found 2250 images belonging to 3 classes.
    Found 750 images belonging to 3 classes.
    epochs=3으로 먼저 테스트
    Epoch 1/3
    47/47 [==============================] - 620s 13s/step - loss: 0.6089 - acc: 0.8347
    
    Epoch 00001: val_loss improved from inf to 0.60892, saving model to best_model_3class.hdf5
    141/141 [==============================] - 11712s 83s/step - loss: 0.9041 - acc: 0.6222 - val_loss: 0.6089 - val_acc: 0.8347
    Epoch 2/3
    47/47 [==============================] - 608s 13s/step - loss: 0.3763 - acc: 0.9053
    
    Epoch 00002: val_loss improved from 0.60892 to 0.37629, saving model to best_model_3class.hdf5
    141/141 [==============================] - 11100s 79s/step - loss: 0.5665 - acc: 0.8360 - val_loss: 0.3763 - val_acc: 0.9053
    Epoch 3/3
    47/47 [==============================] - 622s 13s/step - loss: 0.2596 - acc: 0.9453
    
    Epoch 00003: val_loss improved from 0.37629 to 0.25959, saving model to best_model_3class.hdf5
    141/141 [==============================] - 11173s 79s/step - loss: 0.3948 - acc: 0.8907 - val_loss: 0.2596 - val_acc: 0.9453
    




```python
class_map_3 = train_generator.class_indices
class_map_3
```


    {'bibimbap': 0, 'eggs_benedict': 1, 'grilled_cheese_sandwich': 2}


```python
import matplotlib.pyplot as plt
def plot_accuracy(history, title):
    plt.title(title)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
    plt.show()
def plot_loss(history,title):
    plt.title(title)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'validation_loss'], loc='best')
    plt.show()


plot_accuracy(history,'FOOD101-Inceptionv3')
plot_loss(history,'FOOD101-Inceptionv3')
```

![output_35_0](https://user-images.githubusercontent.com/50260643/62836857-a5915600-bca2-11e9-8160-673d76e467d8.png)



![output_35_1](https://user-images.githubusercontent.com/50260643/62836843-84c90080-bca2-11e9-84e6-d4e9ce04c18c.png)


epochs이 3 밖에 되지 않기에 아쉽지만 accuracy 94%까지
- 해당 모델 best model로 save
 
validation acuuracy가 trainin accuracy보다 더 좋은 정확도를 가지는 이유는
- used a pretrained model trained on ImageNet which contains data from a variety of classes Using dropout can lead to a higher validation accuracy





#### 웹 크롤링을 통해 얻은 이미지를 활용하여 학습된 모델을 활용하여 예측해보기

```python
# Loading the best saved model to make predictions
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model

K.clear_session()

#저장한 best_model 활용
model_best = load_model('best_model_3class.hdf5', compile = False)
```


```python
food_list
```




    ['bibimbap', 'eggs_benedict', 'grilled_cheese_sandwich']



Setting compile=False와 clearing the sessiondms은 저장된 모델을 더 빨리 로딩하고
위와 같은 추가적인 작업없이 더 빠르게 작업 가능


```python
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

def predict_class(model, images, show = True):
  for img in images:
    img = image.load_img(img, target_size=(299, 299))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255.                                      

    pred = model.predict(img)
    index = np.argmax(pred)  # index가 argmax이기 때문에 학습시킨 food classes내에서 prediction
    food_list.sort()
    pred_value = food_list[index]
    if show:
        plt.imshow(img[0])                           
        plt.axis('off')
        plt.title(pred_value)
        plt.show()
```


```python
# 여기서 쓸 푸드 이미지는 직접 다운로드 받은 파일로 사용
images = []
images.append('food-101/test_image/grilled-cheese-sandwich.jpg')
images.append('food-101/test_image/eggs-benedict.jpg')
images.append('food-101/test_image/bibimbap.jpg')
predict_class(model_best, images, True)
```


![output_31_0](https://user-images.githubusercontent.com/50260643/62836964-66640480-bca4-11e9-8dae-cf4d5c87788d.png)


![output_31_1](https://user-images.githubusercontent.com/50260643/62836969-767be400-bca4-11e9-9133-ad979a0da21e.png)


![output_31_2](https://user-images.githubusercontent.com/50260643/62836971-84316980-bca4-11e9-8750-9e54bbca31ac.png)


이미지 순서를 반대로 넣어봤지만 알맞게 예측.
학습을 많이 시키진 못했지만, 클래스 수가 적어서 쉽게 성공.

### Inceptionv3 model을 활용하기 위한 푸드 클래스 11개 활용

위에서 3개의 클래스를 활용한 테스트를 진행했고 예측이 어렵지 않았지만
과연 101개의 클래스를 위한 혹은 50개의 클래스에서도 비슷하게 적용이 될지 의문.

** Even with fine tuning using a pre-trained model, each epoch was taking more than an hour when all 101 classes of data is used(tried this on both Colab and on a Deep Learning VM instance with P100 GPU on GCP)**

대신 같은 모델을 11개의 랜덤 선택된 클래스를 이용하여 성능 체크.

죄송하게도, 면접날까지 11개의 클래스를 이용하여 epochs 2이상을 돌리기에도 면접 날짜를 지나기에 이 부분은 나와있는 데이터를 일부 활용하겠습니다.


```python
# 여기서부터는 랜덤 푸드 클래스를 테스트하기 위한,
import random
def pick_n_random_classes(n):
  food_list = []
  random_food_indices = random.sample(range(len(foods_sorted)),n) # We are picking n random food classes
  for i in random_food_indices:
    food_list.append(foods_sorted[i])
  food_list.sort()
  print("These are the randomly picked food classes we will be training the model on...\n", food_list)
  return food_list
```


```python
# 11개 테스트
n = 11
food_list = pick_n_random_classes(11)
```

    These are the randomly picked food classes we will be training the model on...
     ['apple_pie', 'baby_back_ribs', 'bread_pudding', 'chicken_curry', 'chicken_quesadilla', 'chocolate_cake', 'donuts', 'eggs_benedict', 'garlic_bread', 'gnocchi', 'grilled_cheese_sandwich']
    


```python
# 11개의 mini set 생성
print("Creating training data folder with new classes...")
dataset_mini(food_list, src_train, dest_train)
```

    Creating training data folder with new classes...
    Copying images into apple_pie
    Copying images into baby_back_ribs
    Copying images into bread_pudding
    Copying images into chicken_curry
    Copying images into chicken_quesadilla
    Copying images into chocolate_cake
    Copying images into donuts
    Copying images into eggs_benedict
    Copying images into garlic_bread
    Copying images into gnocchi
    Copying images into grilled_cheese_sandwich
    


```python
print("Creating test data folder with new classes")
dataset_mini(food_list, src_test, dest_test)
```

    Creating test data folder with new classes
    Copying images into apple_pie
    Copying images into baby_back_ribs
    Copying images into bread_pudding
    Copying images into chicken_curry
    Copying images into chicken_quesadilla
    Copying images into chocolate_cake
    Copying images into donuts
    Copying images into eggs_benedict
    Copying images into garlic_bread
    Copying images into gnocchi
    Copying images into grilled_cheese_sandwich
    


```python
# 갯수 파악 'food-101/train', 'food-101/test'
print("Total number of samples in train folder")
train_file_count = sum(len(files) for _, _, files in os.walk(r'food-101/train_mini'))
print(train_file_count)

print("Total number of samples in test folder")
test_file_count = sum(len(files) for _, _, files in os.walk(r'food-101/test_mini'))
print(test_file_count)
```

    Total number of samples in train folder
    8250
    Total number of samples in test folder
    2750
    

```python
# Let's use a pretrained Inceptionv3 model on subset of data with 11 food classes
n = 11

# 시간 측정
import time
startTime = time.time()
endTime = time.time() - startTime
print(endTime)
K.clear_session()

n_classes = n  # 11
img_width, img_height = 299, 299
train_data_dir = 'food-101/train_mini'
validation_data_dir = 'food-101/test_mini'
nb_train_samples = train_file_count  # 8250 
nb_validation_samples = test_file_count  # 2750
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


inception = InceptionV3(weights='imagenet', include_top=False)
x = inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)

predictions = Dense(n, kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)  # 11

model = Model(inputs=inception.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='best_model_11class.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history_11class.log')

history_11class = model.fit_generator(train_generator,
                    steps_per_epoch = nb_train_samples // batch_size,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    epochs=10,   # 적어도 15 - 30정도 하는 것이 좋은 결과.
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])

model.save('model_trained_11class.hdf5')

endTime = time.time() - startTime
print(endTime)

# batch=16 일 때, 1 epoch에 대략 15시간 정도 걸릴 듯
```
**
죄송하게도, 면접날까지 11개의 클래스를 이용하여 epochs 2이상을 돌리기에도 면접 날짜를 지나기에 이 부분은 나와있는 데이터를 일부 활용하겠습니다.**



Found 8250 images belonging to 11 classes.
Found 2750 images belonging to 11 classes.
Epoch 1/10
514/515============================>.] - ETA: 0s - loss: 2.1653 - acc: 0.3193
Epoch 00001: val_loss improved from inf to 1.58121, saving model to best_model_11class.hdf5
515/515==============================] - 419s 813ms/step - loss: 2.1648 - acc: 0.3193 - val_loss: 1.5812 - val_acc: 0.6177
Epoch 2/10
514/515============================>.] - ETA: 0s - loss: 1.4690 - acc: 0.5916
Epoch 00002: val_loss improved from 1.58121 to 0.99604, saving model to best_model_11class.hdf5
515/515==============================] - 347s 674ms/step - loss: 1.4694 - acc: 0.5916 - val_loss: 0.9960 - val_acc: 0.7657
Epoch 3/10
514/515============================>.] - ETA: 0s - loss: 1.1073 - acc: 0.6981
Epoch 00003: val_loss improved from 0.99604 to 0.72963, saving model to best_model_11class.hdf5
515/515==============================] - 347s 673ms/step - loss: 1.1065 - acc: 0.6982 - val_loss: 0.7296 - val_acc: 0.8337
Epoch 4/10
514/515============================>.] - ETA: 0s - loss: 0.9269 - acc: 0.7463
Epoch 00004: val_loss improved from 0.72963 to 0.60849, saving model to best_model_11class.hdf5
515/515==============================] - 347s 673ms/step - loss: 0.9266 - acc: 0.7464 - val_loss: 0.6085 - val_acc: 0.8640
Epoch 5/10
514/515============================>.] - ETA: 0s - loss: 0.8109 - acc: 0.7812
Epoch 00005: val_loss improved from 0.60849 to 0.53972, saving model to best_model_11class.hdf5
515/515==============================] - 347s 673ms/step - loss: 0.8110 - acc: 0.7813 - val_loss: 0.5397 - val_acc: 0.8794
Epoch 6/10
514/515============================>.] - ETA: 0s - loss: 0.7360 - acc: 0.8077
Epoch 00006: val_loss improved from 0.53972 to 0.49090, saving model to best_model_11class.hdf5
515/515==============================] - 346s 673ms/step - loss: 0.7362 - acc: 0.8075 - val_loss: 0.4909 - val_acc: 0.8907
Epoch 7/10
514/515============================>.] - ETA: 0s - loss: 0.6672 - acc: 0.8292
Epoch 00007: val_loss improved from 0.49090 to 0.45883, saving model to best_model_11class.hdf5
515/515==============================] - 347s 673ms/step - loss: 0.6669 - acc: 0.8292 - val_loss: 0.4588 - val_acc: 0.8962
Epoch 8/10
514/515============================>.] - ETA: 0s - loss: 0.6169 - acc: 0.8435
Epoch 00008: val_loss improved from 0.45883 to 0.43328, saving model to best_model_11class.hdf5
515/515==============================] - 348s 675ms/step - loss: 0.6171 - acc: 0.8436 - val_loss: 0.4333 - val_acc: 0.9031
Epoch 9/10
514/515============================>.] - ETA: 0s - loss: 0.5870 - acc: 0.8486
Epoch 00009: val_loss improved from 0.43328 to 0.43104, saving model to best_model_11class.hdf5
515/515==============================] - 345s 670ms/step - loss: 0.5876 - acc: 0.8484 - val_loss: 0.4310 - val_acc: 0.9006
Epoch 10/10
514/515============================>.] - ETA: 0s - loss: 0.5546 - acc: 0.8621
Epoch 00010: val_loss improved from 0.43104 to 0.41282, saving model to best_model_11class.hdf5
515/515==============================] - 344s 667ms/step - loss: 0.5548 - acc: 0.8620 - val_loss: 0.4128 - val_acc: 0.9068
```python
plot_accuracy(history_11class,'FOOD101-Inceptionv3')
plot_loss(history_11class,'FOOD101-Inceptionv3')
```

![11class_acc](https://user-images.githubusercontent.com/50260643/62837160-26525100-bca7-11e9-81f8-f6bed4d14789.png)

![11class_loss](https://user-images.githubusercontent.com/50260643/62837165-38cc8a80-bca7-11e9-893a-6f26ec68881d.png)



epochs이 10일 때, 여전히 loss는 감소하고, 정확도는 올라감


```python
# Loading the best saved model to make predictions
from tensorflow.keras.models import load_model
K.clear_session()
model_best = load_model('best_model_11class.hdf5',compile = False)
```


```python
# Make a list of downloaded images and test the trained model
images = []
images.append('food-101/test_image/apple_pie.jpg')
images.append('food-101/test_image/baby_back_ribs')
images.append('food-101/test_image/bread_pudding')
images.append('food-101/test_image/chicken_curry')
predict_class(model_best, images, True)
```

### 예측한 위의 4가지 모두 맞췄습니다. 



## Feature visualization
Using feature visualization, we can know what a neural network layer and its features are looking for Using attribution, we can understand how the features impact the output and what regions in the image led the model to the generated output

#### Let us now check the model we trained and understand how it sees and classifies


```python
# Load the saved model trained with 3 classes
# 푸드 클래스 3개로 학습한 모델을 활용
K.clear_session()
print("Loading the model..")
model = load_model('best_model_3class.hdf5',compile = False)
print("끝!")
```

    Loading the model..
    끝!
    

Summary of the model gives us the list of all the layers in the network along with other useful details


```python
model.summary()
```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            (None, None, None, 3 0                                            
    __________________________________________________________________________________________________
    conv2d (Conv2D)                 (None, None, None, 3 864         input_1[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization_v1 (BatchNo (None, None, None, 3 96          conv2d[0][0]                     
    __________________________________________________________________________________________________
    activation (Activation)         (None, None, None, 3 0           batch_normalization_v1[0][0]     
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, None, None, 3 9216        activation[0][0]                 
    
    .
    .
    .
    .
    .
    .
    .
    .
    .
    .
    .
   __________________________________________________________________________________________________
    global_average_pooling2d (Globa (None, 2048)         0           mixed10[0][0]                    
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 128)          262272      global_average_pooling2d[0][0]   
    __________________________________________________________________________________________________
    dropout (Dropout)               (None, 128)          0           dense[0][0]                      
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 3)            387         dropout[0][0]                    
    ==================================================================================================
    Total params: 22,065,443
    Trainable params: 22,031,011
    Non-trainable params: 34,432
    __________________________________________________________________________________________________
    


```python
# Defining some helper functions
# 이미지 정제 
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x
```


```python
def generate_pattern(layer_name, filter_index, size=150):
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered.
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]

    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])
    
    # We start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # Run gradient ascent for 40 steps
    step = 1
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)
```


```python
# activation 관련
def get_activations(img, model_activations):
    img = image.load_img(img, target_size=(299, 299))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255
    plt.imshow(img[0])
    plt.show()
    return model_activations.predict(img)
```


```python
def show_activations(activations, layer_names):
    
    images_per_row = 16

    # Now let's display our feature maps
    for layer_name, layer_activation in zip(layer_names, activations):
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1]

        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]

        # We will tile the activation channels in this matrix
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 :, :,
                                                 col * images_per_row + row]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image

        # Display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()
```

Check how many layers are in the trained model(this includes the 1st input layer as well)


```python
len(model.layers)
```




    315




315개를 모두 활용하여 outputs을 시각화 할 수 있지만,
몇 개만 선택해서 시각화.


```python
# We start with index 1 instead of 0, as input layer is at index 0
from tensorflow.keras import models
layers = [layer.output for layer in model.layers[1:11]]
# We now initialize a model which takes an input and outputs the above chosen layers
activations_output = models.Model(inputs=model.input, outputs=layers)
```

the 10 chosen layers contain 3 convolution, 3 batch normalization, 3 activation and 1 max pooling layers


```python
layers
```




    [<tf.Tensor 'conv2d/Conv2D:0' shape=(?, ?, ?, 32) dtype=float32>,
     <tf.Tensor 'batch_normalization_v1/cond/Merge:0' shape=(?, ?, ?, 32) dtype=float32>,
     <tf.Tensor 'activation/Relu:0' shape=(?, ?, ?, 32) dtype=float32>,
     <tf.Tensor 'conv2d_1/Conv2D:0' shape=(?, ?, ?, 32) dtype=float32>,
     <tf.Tensor 'batch_normalization_v1_1/cond/Merge:0' shape=(?, ?, ?, 32) dtype=float32>,
     <tf.Tensor 'activation_1/Relu:0' shape=(?, ?, ?, 32) dtype=float32>,
     <tf.Tensor 'conv2d_2/Conv2D:0' shape=(?, ?, ?, 64) dtype=float32>,
     <tf.Tensor 'batch_normalization_v1_2/cond/Merge:0' shape=(?, ?, ?, 64) dtype=float32>,
     <tf.Tensor 'activation_2/Relu:0' shape=(?, ?, ?, 64) dtype=float32>,
     <tf.Tensor 'max_pooling2d/MaxPool:0' shape=(?, ?, ?, 64) dtype=float32>]




```python
# Get the names of all the selected layers
layer_names = []
for layer in model.layers[1:11]:
    layer_names.append(layer.name)
print(layer_names)
```

    ['conv2d', 'batch_normalization_v1', 'activation', 'conv2d_1', 'batch_normalization_v1_1', 'activation_1', 'conv2d_2', 'batch_normalization_v1_2', 'activation_2', 'max_pooling2d']
    


```python
food = 'food-101/test_image/eggs-benedict.jpg'
activations = get_activations(food,activations_output)
```



![output_66_0](https://user-images.githubusercontent.com/50260643/62837536-89de7d80-bcab-11e9-999c-225759ebde9f.png)



```python
show_activations(activations, layer_names)
```


![output_67_0](https://user-images.githubusercontent.com/50260643/62837562-ca3dfb80-bcab-11e9-93e1-b2e926f9f621.png)



![output_67_1](https://user-images.githubusercontent.com/50260643/62837565-d629bd80-bcab-11e9-9c90-c1c9ff23f509.png)


![output_67_2](https://user-images.githubusercontent.com/50260643/62837556-bb574900-bcab-11e9-8b51-d1a8216029cd.png)


![output_67_3](https://user-images.githubusercontent.com/50260643/62837569-e6419d00-bcab-11e9-81e7-bf08a4085385.png)

![output_67_4](https://user-images.githubusercontent.com/50260643/62837573-f3f72280-bcab-11e9-9df7-5b8a659ad2e5.png)

![output_67_5](https://user-images.githubusercontent.com/50260643/62837578-01aca800-bcac-11e9-9ca9-7e163032d974.png)

![output_67_6](https://user-images.githubusercontent.com/50260643/62837581-0ffac400-bcac-11e9-8a24-56244b1c846d.png)

![output_67_7](https://user-images.githubusercontent.com/50260643/62837585-1e48e000-bcac-11e9-98ef-8203713c4c99.png)

![output_67_8](https://user-images.githubusercontent.com/50260643/62837590-2bfe6580-bcac-11e9-9602-9863f7fbf56d.png)

![output_67_9](https://user-images.githubusercontent.com/50260643/62837598-3d477200-bcac-11e9-93c5-7130fe0ab3d5.png)

What we see in the above plots are the activations or the outputs of each of the 11 layers we chose
The activations or the outputs from the 1st layer(conv2d_1) don't lose much information of the original input
They are the results of applying several edge detecting filters on the input image
With each added layer, the activations lose visual/input information and keeps building on the class/ouput information
As the depth increases, the layers activations become less visually interpretabale and more abstract
By doing so, they learn to detect more specific features of the class rather than just edges and curves
We plotted just 10 out of 314 intermediate layers. We already have in these few layers, activations which are blank/sparse(for ex: the 2 blank activations in the layer activation_1)
These blank/sparse activations are caused when any of the filters used in that layer didn't find a matching pattern in the input given to it
By plotting more layers(specially those towards the end of the network), we can observe more of these sparse activations and how the layers get more abstract

Get the activations for a different input / food






The feature maps in the above activations are for a different input image
We see the same patterns discussed for the previous input image
It is interesting to see the blank/sparse activations in the same layer(activation_1) and for same filters when a different image is passed to the network
Remember we used a pretrained Inceptionv3 model. All the filters that are used in different layers come from this pretrained model

### Look into the sparse activations in the layer activation_1

```python
layer_names
```

    ['conv2d',
     'batch_normalization_v1',
     'activation',
     'conv2d_1',
     'batch_normalization_v1_1',
     'activation_1',
     'conv2d_2',
     'batch_normalization_v1_2',
     'activation_2',
     'max_pooling2d']

We have two blank/sparse activations in layer 6
Below cell displays one of the sparse activations


```python
# Get the index of activation_1 layer which has sparse activations
ind = layer_names.index('activation_1')
sparse_activation = activations[ind]
a = sparse_activation[0, :, :, 13]
a
```




    array([[nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           ...,
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan]], dtype=float32)




```python
all (np.isnan(a[j][k])  for j in range(a.shape[0]) for k in range(a.shape[1]))
```




    True



We can see that the activation has all nan values(it was all zeros when executed outside Kaggle, Im yet to figure out why its showing all nan values here
To know why we have all zero/nan values for this activation, lets visualize the activation at same index 13 from previous layer


```python
# Get the index of batch_normalization_1 layer which has sparse activations
ind = layer_names.index('batch_normalization_v1_1')
sparse_activation = activations[ind]
b = sparse_activation[0, :, :, 13]
b
```




    array([[77.25354 , 77.25354 , 77.25354 , ..., 77.25354 , 77.25354 ,
            77.25354 ],
           [77.25354 , 77.25354 , 77.25354 , ..., 77.25354 , 77.25354 ,
            77.25354 ],
           [77.25354 , 77.25354 , 77.25354 , ..., 77.25354 , 77.25354 ,
            77.25354 ],
           ...,
           [73.993546, 70.73355 , 70.73355 , ..., 77.25354 , 77.25354 ,
            77.25354 ],
           [67.47356 , 64.21356 , 60.953568, ..., 73.993546, 73.993546,
            73.993546],
           [64.21356 , 67.47356 , 67.47356 , ..., 77.25354 , 73.993546,
            77.25354 ]], dtype=float32)



All the values in the above activation map from the layer batch_normalization_1 are negative
This activation in batch_normalization_1 is passed to the next layer activation_1 as input
As the name says, activation_1 is an activation layer and ReLu is the activation function used
ReLu takes an input value, returns 0 if its negative, the value otherwise
Since the input to activation array contains all negative values, the activation layer fills its activation map with all zeros for the index
Now we know why we have those 2 sparse activations in activation_1 layer

### Show the activation outputs of 1st, 2nd and 3rd Conv2D layer activations to compare how layers get abstract with depth


```python
first_convlayer_activation = activations[0]
second_convlayer_activation = activations[3]
third_convlayer_activation = activations[6]
f,ax = plt.subplots(1,3, figsize=(10,10))
ax[0].imshow(first_convlayer_activation[0, :, :, 3], cmap='viridis')
ax[0].axis('OFF')
ax[0].set_title('Conv2d_1')
ax[1].imshow(second_convlayer_activation[0, :, :, 3], cmap='viridis')
ax[1].axis('OFF')
ax[1].set_title('Conv2d_2')
ax[2].imshow(third_convlayer_activation[0, :, :, 3], cmap='viridis')
ax[2].axis('OFF')
ax[2].set_title('Conv2d_3')
```



    Text(0.5, 1.0, 'Conv2d_3')


![output_78_1](https://user-images.githubusercontent.com/50260643/62837653-adee8e80-bcac-11e9-9ab4-b64889ca9e65.png)


# Attribution

So far we were doing activation maps visualization
This helps in understanding how the input is transformed from one layer to another as it goes through several operations
At the end of training, we want the model to classify or detect objects based on features which are specific to the class
For example, when training a dogs vs cats classifier, the model should detect dogs based on features relevant to dog but not cats
To validate how model attributes the features to class output, we can generate heat maps using gradients to find out which regions in the input images were instrumental in determining the class


```python
def get_attribution(food):
    img = image.load_img(food, target_size=(299, 299))
    img = image.img_to_array(img) 
    img /= 255. 
    f,ax = plt.subplots(1,3, figsize=(15,15))
    ax[0].imshow(img)
    
    img = np.expand_dims(img, axis=0) 
    
    preds = model.predict(img)
    class_id = np.argmax(preds[0])
    ax[0].set_title("Input Image")
    class_output = model.output[:, class_id]
    last_conv_layer = model.get_layer("mixed10")
    
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img])
    for i in range(2048):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    ax[1].imshow(heatmap)
    ax[1].set_title("Heat map")
    
    
    act_img = cv2.imread(food)
    heatmap = cv2.resize(heatmap, (act_img.shape[1], act_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(act_img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite('classactivation.png', superimposed)
    img_act = image.load_img('classactivation.png', target_size=(299, 299))
    ax[2].imshow(img_act)
    ax[2].set_title("Class Activation")
    plt.show()
    return preds
```


```python
print("Showing the class map..")
print(class_map_3)
```

    Showing the class map..
    {'bibimbap': 0, 'eggs_benedict': 1, 'grilled_cheese_sandwich': 2}
    


```python
import cv2
pred = get_attribution('food-101/test_image/bibimbap.jpg')
print("Here are softmax predictions..",pred)
```


![output_83_0](https://user-images.githubusercontent.com/50260643/62837665-dffff080-bcac-11e9-8a5f-dac0ff16a208.png)



    Here are softmax predictions.. [[0.6392082  0.04961831 0.31117347]]


약 64%로 조금은 낮은 수치
```python
pred = get_attribution('food-101/test_image/eggs-benedict.jpg')
print("Here are softmax predictions..",pred)
```


![output_86_0](https://user-images.githubusercontent.com/50260643/62837673-fc039200-bcac-11e9-82bb-4c7b6c6b39f7.png)



    Here are softmax predictions.. [[0.01472849 0.96348065 0.02179079]]
    
96%의 정확도

```python
pred = get_attribution('food-101/test_image/grilled-cheese-sandwich.jpg')
print("Here are softmax predictions..",pred)
```


![output_87_0](https://user-images.githubusercontent.com/50260643/62837679-0c1b7180-bcad-11e9-8f9d-9bbe48f37ae9.png)



    Here are softmax predictions.. [[0.0608012  0.09768793 0.8415109 ]]
    
84%의 정확도
heat map이 다른 이미지에 어떻게 다르게 특징을 잡는지 확인할 수 있으며 각각의 이미지를 분류

### 조금 다른 방법으로 멀티 이미지를 분류하기


```python
food = 'food-101/test_image/two-sandwich-1.jpg'
activations = get_activations(food,activations_output)
```


![output_90_0](https://user-images.githubusercontent.com/50260643/62837732-ea6eba00-bcad-11e9-9c36-1e3de7f0d1e6.png)


이미지의 윗부분 - grilled cheese sandwich,
아랫부분 - club sandwich
모델에서 치즈 샌드위치는 학습 시켰으며, 클럽 샌드위치는 학습 시키지 않은 상태에서
두 물체가 함께 있는 이미지가 들어온다면 어떨까?

```python
show_activations(activations, layer_names)
```

    C:\Users\Jongil Park\AppData\Local\conda\conda\envs\python_study\lib\site-packages\ipykernel_launcher.py:25: RuntimeWarning: invalid value encountered in true_divide
    


![output_91_1](https://user-images.githubusercontent.com/50260643/62837779-9a442780-bcae-11e9-92a2-a3cedad4432c.png)

![output_91_2](https://user-images.githubusercontent.com/50260643/62837793-a8924380-bcae-11e9-83e9-d7f04533d18e.png)

![output_91_3](https://user-images.githubusercontent.com/50260643/62837795-b5169c00-bcae-11e9-9d15-77dd11e69d44.png)

![output_91_4](https://user-images.githubusercontent.com/50260643/62837798-c069c780-bcae-11e9-837f-09743194e1e4.png)

![output_91_5](https://user-images.githubusercontent.com/50260643/62837804-ceb7e380-bcae-11e9-88d3-b92600e7c507.png)

![output_91_6](https://user-images.githubusercontent.com/50260643/62837810-dd9e9600-bcae-11e9-8f90-6434c77e3e8d.png)

![output_91_7](https://user-images.githubusercontent.com/50260643/62837815-e8592b00-bcae-11e9-9585-f6492c89549a.png)

![output_91_8](https://user-images.githubusercontent.com/50260643/62837818-f4dd8380-bcae-11e9-92da-e012f73eaf46.png)

![output_91_9](https://user-images.githubusercontent.com/50260643/62837823-0161dc00-bcaf-11e9-8358-aceeec0228b4.png)

![output_91_10](https://user-images.githubusercontent.com/50260643/62837830-10488e80-bcaf-11e9-812d-d580ec420045.png)


```python
pred = get_attribution('food-101/test_image/two-sandwich-1.jpg')
print("Here are softmax predictions..",pred)
```

![output_92_0](https://user-images.githubusercontent.com/50260643/62837836-22c2c800-bcaf-11e9-9826-7f7302e85faf.png)


    Here are softmax predictions.. [[0.09171534 0.06298628 0.84529835]]
    


```python
food = 'food-101/test_image/two-sandwich-2.jpg'
activations = get_activations(food,activations_output)
```

![output_93_0](https://user-images.githubusercontent.com/50260643/62837847-41c15a00-bcaf-11e9-93b8-1daf04554fbf.png)


```python
pred = get_attribution('food-101/test_image/two-sandwich-2.jpg')
print("Here are softmax predictions..",pred)
```

![output_94_0](https://user-images.githubusercontent.com/50260643/62837850-4e45b280-bcaf-11e9-9aa7-7f093519d4c1.png)


    Here are softmax predictions.. [[0.02157178 0.03664926 0.94177896]]
    


### QUESTION!


왜 멀티 라벨 분류를 학습시키지 안았지만 멀티 클래스 분류가 가능할까? 

### ANSWER¶
Once we train a model, it is also necessary to try it in a way we know it may not work(like giving images with multiple classes of objects in it). This will lead to new observations, insights and maybe new conclusions if not inventions!
Flipping the image vertically has flipped the outputs whereas doing a horizontal flip didnt cause any change in output class
As a next step, we can try with more images, from different classes and check if the same pattern exists

# Further Improvements
Try more augmentation on test images
Fine tune the model on the entire dataset(for a few epochs atleast)
Play with hyper parameters, their values and see how it impacts model performance
There is currently no implementation to handle out of distribution / no class scenario. Can try below methods:
Set a threshold for the class with highest score. When model gives prediction score below the threshold for its top prediction, the prediction can be classified as NO-CLASS / UNSEEN
Add a new class called NO-CLASS, provide data from different classes other than those in the original dataset. This way the model also learns how to classify totally unseen/unrelated data
I am yet to try these methods and not sure about the results
Recently published paper - Rethinking ImageNet Pretraining, claims that training from random initialization instead of using pretrained weights is not only robust but also gives comparable results
Pre-trained models are surely helpful. They save a lot of time and computation. Yet, that shouldn't be the reason to not try to train a model from scratch
Time taking yet productive experiment would be to try and train a model on this dataset from scratch
Do more experiments with Model Interpretability and see what can be observed

# References
Multiclass Food Classification using TensorFlow, Kaggle
부분을 많이 참조하였습니다.
