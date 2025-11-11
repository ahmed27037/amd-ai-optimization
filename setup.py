from setuptools import setup, find_packages

found_packages = find_packages()
packages = ['amd_ai_optimization'] + [f'amd_ai_optimization.{pkg}' for pkg in found_packages if pkg != 'amd_ai_optimization']

package_dir = {'amd_ai_optimization': '.'}
for pkg in found_packages:
    if pkg != 'amd_ai_optimization':
        package_dir[f'amd_ai_optimization.{pkg}'] = pkg

setup(
    name="amd_ai_optimization",
    version="1.0.0",
    description="AI/ML Hardware Optimization Suite",
    packages=packages,
    package_dir=package_dir,
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
    ],
    python_requires=">=3.8",
)

