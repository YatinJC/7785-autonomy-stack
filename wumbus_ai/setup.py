from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'wumbus_ai'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'models'), glob('wumbus_ai/models/*.pth')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yc',
    maintainer_email='ychandar3@gatech.edu',
    description='AI-powered autonomous navigation for TurtleBot3',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sign_detector = wumbus_ai.sign_detector:main',
            'ai_driver = wumbus_ai.ai_driver:main',
            'img_sub = wumbus_ai.img_sub:main',
        ],
    },
)
