# Yatin Chandar and Arun Showry Busani

from setuptools import find_packages, setup

package_name = 'wesuckatcoding_object_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yc',
    maintainer_email='ychandar3@gatech.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'chase_object = wesuckatcoding_object_follower.chase_object:main',
            'detector = wesuckatcoding_object_follower.detect:main',
            'viewer = wesuckatcoding_object_follower.viewer:main',
            'get_object_range = wesuckatcoding_object_follower.get_object_range:main',
        ],
    },
)
