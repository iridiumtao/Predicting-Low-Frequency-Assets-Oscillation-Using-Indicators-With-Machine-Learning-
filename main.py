import tensorflow as tf

def is_gpu_available():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPUs are available.")
        print(gpus)
        return True
    else:
        print("GPUs are not available.")
        return False

if __name__ == '__main__':
    print("hello world")
    is_gpu_available()