from setuptools import find_packages, setup

package_name = 'WESUCKATCODING_object_follower'

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
            'rotate_robot = WESUCKATCODING_object_follower.rotate_robot:main'
            'dectector = WESUCKATCODING_object_follower.dectector:main'
            'viewer = WESUCKATCODING_object_follower.viewer:main'
        ],
    },
)
