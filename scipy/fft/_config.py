from . import _pocketfft as pfft
from . import _fftpack as fftpack

_fallbacks = {
    # General FFTs
    'fft': pfft.fft,
    'ifft': pfft.ifft,
    'fft2': pfft.fft2,
    'ifft2': pfft.ifft2,
    'fftn': pfft.fftn,
    'ifftn': pfft.ifftn,
    # Real FFTs
    'rfft': pfft.rfft,
    'irfft': pfft.irfft,
    'rfft2': pfft.rfft2,
    'irfft2': pfft.irfft2,
    'rfftn': pfft.rfftn,
    'irfftn': pfft.irfftn,
    # DCTs
    'dct': fftpack.dct,
    'idct': fftpack.idct,
    'dctn': fftpack.dctn,
    'idctn': fftpack.idctn,
    # DSTs
    'dst': fftpack.dst,
    'idst': fftpack.idst,
    'dstn': fftpack.dstn,
    'idstn': fftpack.idstn,
}

_functions = _fallbacks.copy()
