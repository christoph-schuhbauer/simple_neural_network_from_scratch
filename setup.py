import pickle
import numpy as np
import os

def normalize(value):

    # value  ( 2 / max value ) - 1
    # scales [0, 255] --> [-1, 1]
    # / 255 --> 256 'cause no real zeros and ones

    max_value = np.max(value)
    max_value = max_value*1.001

    new_value = (value * (2 / max_value)) - 1


    #print(f'new value: {new_value}')

    #new_value = np.round(new_value, 2)
    return new_value


def preprocess_data(data):

    # 255 --> schwarz
    print('yo')
    first_list = data[0]
    sec_list = data[1]

    norm_images = [np.reshape(normalize(x), (784, -1)) for x in first_list]

    one_hot_labels = []

    for x in sec_list:
        one_hot = [0 for x in range(10)]
        one_hot[x] = 1
        one_hot_labels.append(one_hot)

    for x in one_hot_labels:
        print(x)

    data = [norm_images, one_hot_labels]


    with open("data/digital_numbers.pickle", 'wb') as f:
        pickle.dump(data, f)



def prep_mnist():

    # form: (train_imgs, train_imgs, train_labels, test_labels, train_onehot_labels, test_onehot_labels)
    # image are greyscale, values from [0.01, 0.99]
    # training-data: 60k, test-data: 10k

    with open("C:\\Users\\chris\\mnist\\pickled_mnist.pkl", "br") as fh:
        loaded_data = pickle.load(fh)

    #data = (loaded_data[0], loaded_data[1], loaded_data[4], loaded_data[5])
    #print(type(loaded_data[0]))
    trainimages = []

    for x in loaded_data[0]:
        norm = normalize(x)
        trainimages.append(norm)

    tr_i = np.array(trainimages)

    test_imgs = []
    for y in loaded_data[1]:
        norm = normalize(y)
        test_imgs.append(norm)
    te_i = np.array(test_imgs)

    data = (tr_i, te_i, loaded_data[4], loaded_data[5])

    with open("data/mnist.pickle", 'wb') as f:
        pickle.dump(data, f)
    print('finish')
    return data


def prep_digital():


    with open('data/digital_numbers.pickle', 'rb') as handle:
        data = pickle.load(handle)

    #print(len(data[0]))
    images = data[0]
    labels = data[1]
    split = int(len(images) *0.8)
    #print(split)

    train_imgs = []
    test_imgs = []

    train_labels = []
    test_labels = []

    for img, label in zip(images, labels):
        rad_value = np.random.randint(low=1, high=101)
        x = normalize(img)
        if rad_value < 80:

            train_imgs.append(x)
            train_labels.append(label)

        else:
            test_imgs.append(x)
            test_labels.append(label)

    train_imgs = np.array(train_imgs)
    test_imgs = np.array(test_imgs)

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)






    data = (train_imgs, test_imgs, train_labels, test_labels)

    with open("data/digital_numbers_4set.pickle", 'wb') as f:
        pickle.dump(data, f)



def load_digital_numbers():

    with open('data/digital_numbers_4set.pickle', 'rb') as handle:
        data = pickle.load(handle)

    return data



def load_local_mnist():

    with open('data/mnist.pickle', 'rb') as handle:
        data = pickle.load(handle)


    return data


def recreate_img(img_array):

    #img = np.reshape(img_array, (28, 28))
    max_value = 255
    value = ((img_array + 1) * max_value) / 2
    img = np.reshape(value, (28, 28))

    return img


