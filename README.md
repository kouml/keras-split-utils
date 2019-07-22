# keras-split-utils
This is a simple keras train validation split utility.(`split_utils.py`)  
It can useful for `keras.preprocessing.image.ImageDataGenerator` and `flow_from_directory`.

## usage:
1. clone and copy `split_utils.py` to your directory

``` bash
$ git clone git@github.com:kouml/keras-split-utils.git
$ cp keras-split-utils/split_utils.py <your directory>
```

2. you can use `train_valid_split()` like the following snippet.

``` python
import split_utils
original_dir = './data/'
batch_size = 32
validation_split = 0.1

# all data in train_dir which are alias to original_data.(both dir is temporary directory)
# don't clear base_dir, because this directory holds on temp directory.
base_dir, train_dir, val_dir = split_utils.train_valid_split(original_dir, validation_split, seed=1)

# generator for train data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(28, 28),
    batch_size=batch_size,
    color_mode='grayscale'
)

# generator for validation data
val_datagen = ImageDataGenerator(rescale=1./255)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(28, 28),
    batch_size=batch_size,
    color_mode='grayscale'
    )

print('the ratio of validation_split is {}'.format(validation_split))
print('the size of train_dir is {}'.format(train_gen.n))
print('the size of val_dir is {}'.format(val_gen.n))
```

## Etc
`example.py` is a simple example training/validation code with mnist dataset.