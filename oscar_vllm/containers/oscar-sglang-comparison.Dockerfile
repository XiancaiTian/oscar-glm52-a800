FROM docker.m.daocloud.io/lmsysorg/sglang:v0.5.10.post1

LABEL org.opencontainers.image.title="OSCAR official SGLang comparison"
LABEL org.opencontainers.image.revision="41ebcdba3db5f0ce1339c3727caea80df575d437"

COPY python /opt/oscar/sglang-research/python

ENV PYTHONPATH=/opt/oscar/sglang-research/python
WORKDIR /opt/oscar
