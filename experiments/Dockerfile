FROM nvcr.io/nvidia/jax:24.04-py3

ARG USERNAME=myuser
ARG UID=1001
ARG GID=1002

RUN apt-get update && \
    apt-get install -y --no-install-recommends tmux ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Match container user/group to host IDs to avoid permission issues on bind mounts.
RUN if ! getent group "${GID}" >/dev/null; then groupadd --gid "${GID}" "${USERNAME}"; fi && \
    useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash "${USERNAME}"

WORKDIR /home/${USERNAME}

# default workdir
# WORKDIR /home/workdir
# copy the JaxMARL folder to docker

COPY JaxMARL ./JaxMARL
COPY experiments ./overcooked_v2_experiments
RUN chown -R "${UID}:${GID}" /home/${USERNAME}

USER ${USERNAME}
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

RUN pip install --user -e "JaxMARL[algs]"

RUN pip install --user -e overcooked_v2_experiments

# Ensure wandb's generated protobuf bindings can import runtime_version.
RUN pip install --user --upgrade "protobuf>=5.28,<6"

# RUN pip install --user -e overcooked_v2_experiments/overcooked_ai

#for secrets and debug
ENV WANDB_API_KEY=""
ENV WANDB_ENTITY=""
RUN mkdir -p /home/${USERNAME}/runs /home/${USERNAME}/overcooked_v2_experiments/runs && \
    git config --global --add safe.directory /home/${USERNAME}/JaxMARL && \
    git config --global --add safe.directory /home/${USERNAME}/overcooked_v2_experiments

WORKDIR /home/${USERNAME}/overcooked_v2_experiments
