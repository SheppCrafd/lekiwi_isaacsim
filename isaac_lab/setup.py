from setuptools import find_packages, setup

setup(
    name="lekiwi_tasks",
    version="0.1.0",
    packages=find_packages(include=["lekiwi_tasks", "lekiwi_tasks.*"]),
    install_requires=[],  # deliberately empty -- isaaclab/isaacsim/rsl_rl/torch come from the Isaac Sim install itself (Phase 1), pinning versions here would fight that install rather than help it
)
