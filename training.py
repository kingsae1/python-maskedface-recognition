import os
import tensorflow
import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

keras.__version__

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
conv_base.summary()


base_dir = os.path.dirname(os.path.abspath(__file__))

image_path = 'lfw-deepfunneled-result_ttv'
# image_path = 'AFDB_face_dataset_ttv'

train_dir = os.path.join(base_dir, 'images/'+ image_path +'/train')
validation_dir = os.path.join(base_dir, 'images/' + image_path + '/val')
test_dir = os.path.join(base_dir, 'images/' + image_path + '/test')

batchSize = 20
genStep = 50
classLength = len(os.listdir(train_dir))

testCase = [
    { "step" : 100, "epoch" : 10, "valid" : 50 },
    { "step" : 100, "epoch" : 20, "valid" : 50 },
    { "step" : 100, "epoch" : 30, "valid" : 50 },
    { "step" : 100, "epoch" : 40, "valid" : 50 },
    { "step" : 100, "epoch" : 50, "valid" : 50 },
    { "step" : 100, "epoch" : 60, "valid" : 50 },
    { "step" : 100, "epoch" : 70, "valid" : 50 },
    { "step" : 100, "epoch" : 80, "valid" : 50 },
    { "step" : 100, "epoch" : 90, "valid" : 50 },
    { "step" : 100, "epoch" : 100, "valid" : 50 },
    { "step" : 100, "epoch" : 150, "valid" : 50 },
    { "step" : 100, "epoch" : 200, "valid" : 50 },
]
print('ClassLength : ' + str(classLength))

totalResult = '[Log] Total Result'

for options in testCase : 
    stepEpoch = options['step']
    epoch = options['epoch']
    valStep = options['valid']

    train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=0.1,
      horizontal_flip=True,
      fill_mode='nearest')

    # 검증 데이터는 증식되어서는 안 됩니다!
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        # 타깃 디렉터리
        train_dir,
        # 모든 이미지의 크기를 150 × 150로 변경합니다
        target_size=(150, 150),
        batch_size=batchSize,
        # binary_crossentropy 손실을 사용하므로 이진 레이블이 필요합니다
        class_mode='categorical')

    # Batch
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=batchSize,
        class_mode='categorical')

    conv_base.trainable = True

    for layer in conv_base.layers:
        if layer.name.find('block5_conv') > -1:
            print(layer.name)
            layer.trainable = True
            print(layer.trainable)
        else:
            layer.trainable = False
            print(layer.trainable)

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    # model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(classLength, activation='softmax'))
    # model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()

    print('conv_base를 동결한 후 훈련되는 가중치 객체의 수:', 
        len(model.trainable_weights))

    # fit() 메서드의 callbacks 매개변수를 사용하여 원하는 개수만큼 콜백을 모델로 전달
    callback_list = [
        tensorflow.keras.callbacks.EarlyStopping(
            monitor='val_acc', # 모델의 검증 정확도 모니터링
            patience=7, # 1 에포크보다 더 길게 향상되지 않으면 중단
        ),
        tensorflow.keras.callbacks.ModelCheckpoint(
            filepath='myquize1_model.checkpoint', # 저장
            monitor='val_loss',
            save_best_only=True, # 가장 좋은 모델
        )]

    # model.compile(loss='binary_crossentropy',
    #               optimizer=optimizers.RMSprop(learning_rate=1e-4),
    #               metrics=['acc'])

    # optimizerName Adam|RMSprop
    optimizerName = 'Adam' 

    model.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(learning_rate=1e-4),
            # optimizer=optimizers.RMSprop(learning_rate=1e-4),
            metrics=['acc', 'top_k_categorical_accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=stepEpoch,  
        epochs=epoch,         
        validation_data=validation_generator,
        validation_steps=valStep,  
        callbacks=callback_list)

    # 결과치 저장
    pic1 = pd.DataFrame(history.history)[['loss', 'val_loss']].plot(figsize=(8, 6))
    pic2 = pd.DataFrame(history.history)[['acc', 'val_acc']].plot(figsize=(8, 6))
    pic3 = pd.DataFrame(history.history)[['top_k_categorical_accuracy', 'val_top_k_categorical_accuracy']].plot(figsize=(8, 6))

    picFig1 = pic1.get_figure()
    picFig2 = pic2.get_figure()
    picFig3 = pic3.get_figure()

    picPath = 'outputs/step' + str(stepEpoch) + '_ep'+ str(epoch)+'_val'+ str(valStep)

    new_root = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), picPath)

    if not os.path.exists(new_root):
        os.mkdir(new_root)

    picFig1.savefig(picPath + '/output1.png')
    picFig2.savefig(picPath + '/output2.png')
    picFig3.savefig(picPath + '/output3.png')

    test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(150, 150),
            batch_size=batchSize,
            class_mode='categorical')  

    test_loss, test_acc, test_top_k_acc = model.evaluate_generator(test_generator, steps=genStep)

    # test_acc = 0.1
    # test_top_k_acc = 0.1

    infoText = '[LOG] Identity : ' + str(classLength) + ' / Data Path : ' + image_path + ' / Optimizer : ' + optimizerName
    logText = '[LOG] Test Acc : ' + str(test_acc) + ' / Test Top-5 Acc: ' + str(test_top_k_acc) + ' S:' + str(stepEpoch) + '|E:' + str(epoch) + '|V:' + str(valStep) + '|GS:' + str(genStep)

    totalResult += logText + '\n'
    
    with open(picPath +'/log.txt', 'w', encoding='utf-8') as f:
        f.write(f'{infoText}\n{logText}\n')

print(totalResult)