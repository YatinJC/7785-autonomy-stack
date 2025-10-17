from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'wumbus'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yc',
    maintainer_email='ychandar3@gatech.edu',
    description='Wumbus autonomous navigation package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller = wumbus.controller:main',
            'derez_lidar = wumbus.derez_lidar:main',
            'goal_tracker = wumbus.goal_tracker:main',
            'search_node = wumbus.search_node:main',
            'odom_correct = wumbus.odom_correct:main',
        ],
    },
)
