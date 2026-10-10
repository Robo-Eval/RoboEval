# Hugging Face Space (Docker SDK) for the CoRL 2026 RoboEval challenge portal.
# Build context: the contents of portal/ (deploy them to the Space repo root).
FROM python:3.10-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git gcc libc6-dev libgl1 libegl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# SYSTEM=spaces makes Gradio use real Hugging Face sign-in. Docker Spaces do not
# set it, and without it Gradio mocks every login as the HF_TOKEN owner.
# Same pinned numeric environment as roboeval.challenge.determinism, so logged
# actions replay exactly as they ran on the participant's machine.
ENV OPENBLAS_CORETYPE=Haswell \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MKL_CBWR=COMPATIBLE \
    PYTHONUNBUFFERED=1 \
    SYSTEM=spaces

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user

# The validator must be the challenge release participants evaluate with.
ARG ROBOEVAL_REPO=https://github.com/Robo-Eval/RoboEval.git
ARG ROBOEVAL_REF=corl2026-challenge
RUN git clone --branch "$ROBOEVAL_REF" "$ROBOEVAL_REPO" RoboEval \
    && git -C RoboEval submodule update --init \
    && pip install --no-cache-dir --user -e ./RoboEval

COPY --chown=user requirements.txt /home/user/app/requirements.txt
RUN pip install --no-cache-dir --user -r /home/user/app/requirements.txt

COPY --chown=user . /home/user/app
WORKDIR /home/user/app
EXPOSE 7860
CMD ["python", "app.py"]
