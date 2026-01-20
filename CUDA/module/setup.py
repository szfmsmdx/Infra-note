import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

abs_path = os.path.dirname(os.path.abspath(__file__))

setup(
    name='custom_ops_cuda', # 编译后在 Python 里 import 的名字
    ext_modules=[
        CUDAExtension(
            name='custom_ops_cuda', 
            sources=[
                'custom_ops.cpp',        # 桥梁文件
                'csrc/kernel/rms_norm.cu',  # 具体的 Kernel 文件
                'csrc/kernel/gemm.cu'
            ],
            # 如果你有头文件在 include 文件夹里，可以取消注释下面这行
            # include_dirs=[os.path.join(abs_path, 'csrc/include')],
            extra_compile_args={
                'cxx': ['-g'],           # C++ 编译参数
                'nvcc': ['-O3']          # CUDA 编译参数，O3 代表最高等级优化
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)