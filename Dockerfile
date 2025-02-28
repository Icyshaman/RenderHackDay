FROM alpine:latest

WORKDIR /sbot
COPY requirements.txt sbot.py ./

RUN apk update && apk add --no-cache python3 py3-pip py3-virtualenv \
    && apk add --no-cache --virtual .build-deps gcc g++ musl-dev python3-dev \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt \
    && /opt/venv/bin/python -m spacy download en_core_web_sm \
    && rm -rf /var/cache/apk/*

ENV PATH="/opt/venv/bin:$PATH"

CMD ["python", "sbot.py"]