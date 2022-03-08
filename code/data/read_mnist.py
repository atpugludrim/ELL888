import numpy as np

images_path = '../MNIST/train-images-idx3-ubyte'
labels_path = '../MNIST/train-labels-idx1-ubyte'

# *- http://yann.lecun.com/exdb/mnist/ -*
# label file format
# [offset]   [type]         [value]    [description]
# 0000       32 bit int     0x0000801  magic number
# 0004       32 bit int     60000      number of items
# 0008       unsigned byte  ??         label
# 0009       unsigned byte  ??         label
# ...          ...           ...        ...


# image file format
# [offset]   [type]         [value]    [description]
# 0000       32 bit int     0x0000801  magic number
# 0004       32 bit int     60000      number of items
# 0008       32 bit int     28         number of rows
# 0012       32 bit int     28         number of cols
# 0016       unsigned byte  ??         pixel
# 0017       unsigned byte  ??         pixel
# 0018       unsigned byte  ??         pixel
# ...          ...           ...        ...


def read_images(images_path):
    # https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python/53570674
    f = open(images_path,"rb")
    discard = f.read(16)
    buf = f.read(28*28*60000)
    images = np.frombuffer(buf,dtype=np.uint8).astype(np.float32)
    images = images.reshape(60000, 28, 28, 1)
    f.close()
    return images

def read_labels(labels_path):
    f = open(labels_path,"rb")
    discard = f.read(8)
    buf = f.read(60000)
    labels = np.frombuffer(buf,dtype=np.uint8).astype(np.int64)
    f.close()
    return labels

def main():
    import matplotlib.pyplot as plt
    global images_path, labels_path
    images = read_images(images_path)
    labels = read_labels(labels_path)
    idx = np.random.randint(0,60000,(10,))
    fig, ax = plt.subplots(2,5,figsize=(10,4))
    ctr = 0
    for i in range(2):
        for j in range(5):
            ax[i][j].imshow(images[idx[ctr]],cmap='gray')
            ax[i][j].set_title("{}".format(labels[idx[ctr]]))
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            ctr += 1
    plt.show()

if __name__ == "__main__":
    main()
