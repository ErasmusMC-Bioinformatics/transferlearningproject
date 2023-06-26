FROM tensorflow/tensorflow:2.10.1-gpu-jupyter

COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# test gpu
# import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))