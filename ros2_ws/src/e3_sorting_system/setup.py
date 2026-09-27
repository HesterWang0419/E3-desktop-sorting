from glob import glob

from setuptools import find_packages, setup


package_name = "e3_sorting_system"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            "share/" + package_name + "/launch",
            glob("launch/*.launch.py"),
        ),
        (
            "share/" + package_name + "/config",
            glob("config/*.yaml"),
        ),
        (
            "share/" + package_name + "/worlds",
            glob("worlds/*.world"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="eve",
    maintainer_email="chueve17@gmail.com",
    description=(
        "Desktop object detection, fixed-grid sorting, "
        "and task control for experiment E3"
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            (
                "object_detector = "
                "e3_sorting_system.object_detector:main"
            ),
            (
                "pick_action_server = "
                "e3_sorting_system.pick_action_server:main"
            ),
            (
                "task_manager = "
                "e3_sorting_system.task_manager:main"
            ),
        ],
    },
)
