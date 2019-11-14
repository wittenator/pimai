FROM pytorch/pytorch:latest
USER $UID

ENV PATH="/home/me/.local/bin:${PATH}"

RUN /opt/conda/bin/conda install -y nodejs Cython tensorflow pandas scikit-learn matplotlib seaborn jupyter jupyterlab && \
    /opt/conda/bin/conda install -c conda-forge tensorboardx && \
    /opt/conda/bin/conda clean -ya

RUN jupyter labextension install jupyterlab_tensorboard

RUN pip install jupyter_tensorboard torchvision scikit-image

RUN mkdir -p /home/me && chmod 1777 /home/me

ENV HOME /home/me

WORKDIR /workspace

RUN pip install --user poetry

# We copy just the requirements.txt first to leverage Docker cache
COPY ./src/notebook/pyproject.toml ./pyproject.toml
COPY ./src/notebook/poetry.lock ./poetry.lock

#RUN poetry config settings.virtualenvs.create false
RUN poetry config settings.virtualenvs.in-project true
RUN poetry install

# tensorboard
EXPOSE 6006
# jupyter notebook
EXPOSE 8888

COPY ./src/notebook/start_notebook.sh ./start_notebook.sh

CMD jupyter lab --no-browser --ip=0.0.0.0 --port=8888



