import os
SPCONV_ALGO = os.environ.get('SPCONV_ALGO', 'implicit_gemm')
from .conv_trispconv import *
