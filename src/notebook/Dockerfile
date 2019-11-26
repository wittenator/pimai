FROM pytorch/pytorch:latest

USER $UID

ENV PATH="/home/me/.local/bin:${PATH}"

RUN mkdir -p /home/me && chmod 1777 /home/me

ENV HOME="/home/me"


RUN /opt/conda/bin/conda install --use-local -y nodejs Cython tensorflow pandas scikit-learn matplotlib seaborn jupyter jupyterlab && \
    /opt/conda/bin/conda clean -ya

WORKDIR /workspace

RUN pip install --user torchvision scikit-image skorch

# tensorboard
EXPOSE 6006
# jupyter notebook
EXPOSE 8888

RUN chmod -R 1777 /home/me

CMD tensorboard --logdir /data/runs --bind_all & jupyter lab --no-browser --ip=0.0.0.0 --port=8888



