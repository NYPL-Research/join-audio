import nox

# Define sessions
@nox.session(python=["3.11"])
def tests(session):
    """Run the test suite."""
    session.install("poetry")
    session.run("poetry", "install", external=True)
    session.run("pytest", "tests")

@nox.session
def lint(session):
    """Run linting."""
    session.install("flake8")
    session.run("flake8", "src", "tests")

@nox.session
def black(session):
    """Run black code formatter check."""
    session.install("black")
    session.run("black", "--check", "src", "tests")

@nox.session
def isort(session):
    """Run isort import sorter check."""
    session.install("isort")
    session.run("isort", "--check-only", "src", "tests")


@nox.session
def mypy(session):
    """Run static type checking."""
    session.install("mypy", "numpy", "scipy", "soundfile")
    session.run("mypy", "src")
