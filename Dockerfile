FROM tensorflow/tensorflow:2.10.1-gpu-jupyter

RUN python3 -m pip install jupyter jupyterlab




# import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))