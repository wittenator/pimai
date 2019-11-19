FROM pytorch/pytorch:latest

ENV PATH="/home/me/.local/bin:${PATH}"

USER $UID

RUN mkdir -p /home/me && chmod 1777 /home/me

ENV HOME /home/me

RUN /opt/conda/bin/conda install --use-local -y nodejs Cython tensorflow pandas scikit-learn matplotlib seaborn jupyter jupyterlab && \
    /opt/conda/bin/conda install --use-local -c conda-forge tensorboardx && \
    /opt/conda/bin/conda clean -ya

RUN jupyter labextension install jupyterlab_tensorboard

RUN pip install --user jupyter_tensorboard torchvision scikit-image

WORKDIR /workspace

RUN pip install --user poetry

# We copy just the requirements.txt first to leverage Docker cache
COPY ./src/notebook/pyproject.toml ./pyproject.toml
COPY ./src/notebook/poetry.lock ./poetry.lock

RUN poetry config settings.virtualenvs.create false
RUN poetry config settings.virtualenvs.in-project true
RUN poetry install

# tensorboard
EXPOSE 6006
# jupyter notebook
EXPOSE 8888

RUN chmod -R 1777 /home/me

RUN jupyter tensorboard enable --user
CMD tensorboard --logdir ./src/runs --bind_all & jupyter lab --no-browser --ip=0.0.0.0 --port=8888



