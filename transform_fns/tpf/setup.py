from Cython.Build import cythonize
from setuptools import Extension, setup
# from setuptools.command.build_ext import build_ext


# Avoid a gcc warning below:
#   cc1plus: warning: command line option '-Wstrict-prototypes is valid
#   for C/ObjC but not for C++
# class BuildExt(build_ext):
#     def build_extensions(self):
#         self.compiler.compiler_so.remove('-Wstrict-prototypes')
#         super(BuildExt, self).build_extensions()


exts = [
    Extension(
        'tpf_pyx',
        sources=['tpf_pyx.pyx'],
        extra_compile_args=["-O3"],
        language="c++",
    )
]

ext_modules = cythonize(exts, compiler_directives={'language_level': "3"})

setup(
    # cmdclass={'build_ext': BuildExt},
    name='tpf',
    packages=['tpf'],
    package_dir={'tpf': 'tpf'},
    install_requires=[
        'cython>=0.29',
    ],
    version='1.0',
    ext_modules=ext_modules
)
