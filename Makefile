UID = $(shell id -u)
GID = $(shell id -g)

docker-exec = docker compose exec app /bin/bash -c "$1"

up: compose
.PHONY: up

compose:
	docker compose up -d
.PHONY: compose

build: halt
	docker compose build
.PHONY: build

halt:
	docker compose stop
.PHONY: halt

destroy:
	docker compose down --remove-orphans --volumes
.PHONY: destroy

ssh:
	docker compose exec app /bin/bash
.PHONY: ssh

provision:
	$(call docker-exec, poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR)
.PHONY: provision
