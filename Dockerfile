FROM python:3.11-slim

# git: pyAgxArm을 GitHub에서 설치
# fonts-nanum: matplotlib 한글 라벨
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        fonts-nanum \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache/matplotlib

ENV MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN grep -v '^PyYAML' requirements.txt > req-docker.txt \
    && pip install --no-cache-dir -r req-docker.txt

COPY src ./src

CMD ["bash"]
