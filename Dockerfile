FROM python:3.11-buster as runtime

ENV VIRTUAL_ENV=/usr/app/.venv \
    PATH="/usr/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y git

RUN pip install poetry==1.4.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# A directory to have app data
WORKDIR /usr/app

COPY pyproject.toml .
COPY poetry.lock .

RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

CMD ["streamlit", "run", "streamlit_agent/chat_with_sitemap.py", "--server.port", "8051"]
