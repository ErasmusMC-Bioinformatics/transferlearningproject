FROM ghcr.io/mamba-org/micromamba

COPY --chown=$MAMBA_USER:$MAMBA_USER transferlearning.yaml /tmp/env.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt /tmp/requirements.txt
RUN micromamba install -y -n base -f /tmp/env.yaml && micromamba clean --all --yes

# import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))