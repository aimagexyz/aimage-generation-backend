# AI Mage Supervision Backend

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyPI implementation](https://img.shields.io/pypi/implementation/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![Python 3.11.9](https://img.shields.io/badge/python-3.11.9-blue.svg)](https://www.python.org/downloads/release/python-3119/)


This is the backend of the AI Mage project. It is a RESTful API that provides endpoints for the frontend to interact with the database.

## Tech Stack
- Python>=3.11
- uvicorn
- FastAPI
- Tortoise-ORM
- PostgreSQL
- vecs

## Preparation before project start

1. Install Python 3.11 from the [official website](https://www.python.org/downloads/)
2. Clone this repo, and `cd` to the repo root directory path
3. Install `virtualenv` library (Use `pip install virtualenv`)
4. Run `virtualenv venv`
5. Use `vscode` open the repo from the root directory path of the repo
6. Select the newly created `venv` environment
7. Create a new terminal in `vscode`, you should see the word `(venv)` in front of the command
8. Install project required dependencies, use `pip install -r requirements.txt`
9. Place the environment file `.env` in the repo root directory (You can ask the project administrator for the environment file)
10. Make sure the above goes well, you should now be able to run the project directly

## Git Pre Commit Hooks
- Config in `.pre-commit-config.yaml`
  - Using `autopep8` to cleanup code style
  - Using `mypy` to ensure type safety
  - Using `pyright` to ensure type safety
- Run `pre-commit run --all-files` to auto fix all code styles, and get mypy type checking reports
- If Git Pre Commit Hooks fails, please fix the git settings and related code first, and **do not force commit code**!

## Run

Make sure you are currently in the `venv` environment created by the above preparations and just run:

```bash
uvicorn aimage_supervision.app:app --reload
# or
python main.py
```

## Database Operations (TortoiseORM with aerich)

- Init ORM: `aerich init -t aimage_supervision.settings.TORTOISE_ORM` (Only first, no need)
- Init database: `aerich init-db` (Only first, no need)
- Make migrations: `aerich migrate`
- Migrate(to DB): `aerich upgrade`

## Project Specifications

- Comply with the paradigm specified by [`PEP8`](https://peps.python.org/pep-0008/)
- Develop with `vscode` and the `pylance` plugin
- Code formatting with [`autopep8`](https://pypi.org/project/autopep8/)
