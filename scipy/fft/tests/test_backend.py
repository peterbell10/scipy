import numpy as np
import scipy.fft
from scipy.fft import set_backend, backend
import scipy.fft._fftpack as fftpack
import scipy.fft.tests.mock_backend as mock

from numpy.testing import assert_allclose, assert_equal, assert_
import pytest

fnames = ('fft', 'fft2', 'fftn',
          'ifft', 'ifft2', 'ifftn',
          'rfft', 'rfft2', 'rfftn',
          'irfft', 'irfft2', 'irfftn',
          'dct', 'idct', 'dctn', 'idctn',
          'dst', 'idst', 'dstn', 'idstn')

np_funcs = (np.fft.fft, np.fft.fft2, np.fft.fftn,
            np.fft.ifft, np.fft.ifft2, np.fft.ifftn,
            np.fft.rfft, np.fft.rfft2, np.fft.rfftn,
            np.fft.irfft, np.fft.irfft2, np.fft.irfftn,
            fftpack.dct, fftpack.idct, fftpack.dctn, fftpack.idctn,
            fftpack.dst, fftpack.idst, fftpack.dstn, fftpack.idstn)

funcs = (scipy.fft.fft, scipy.fft.fft2, scipy.fft.fftn,
         scipy.fft.ifft, scipy.fft.ifft2, scipy.fft.ifftn,
         scipy.fft.rfft, scipy.fft.rfft2, scipy.fft.rfftn,
         scipy.fft.irfft, scipy.fft.irfft2, scipy.fft.irfftn,
         scipy.fft.dct, scipy.fft.idct, scipy.fft.dctn, scipy.fft.idctn,
         scipy.fft.dst, scipy.fft.idst, scipy.fft.dstn, scipy.fft.idstn)

mocks = (mock.fft, mock.fft2, mock.fftn,
         mock.ifft, mock.ifft2, mock.ifftn,
         mock.rfft, mock.rfft2, mock.rfftn,
         mock.irfft, mock.irfft2, mock.irfftn,
         mock.dct, mock.idct, mock.dctn, mock.idctn,
         mock.dst, mock.idst, mock.dstn, mock.idstn)


@pytest.mark.parametrize("func, np_func, mock", zip(funcs, np_funcs, mocks))
def test_backend_call(func, np_func, mock):
    x = np.arange(20).reshape((10,2))
    answer = np_func(x)
    assert_allclose(func(x), answer, atol=1e-10)

    with backend('module://scipy.fft.tests.mock_backend', on_missing='raise'):
        mock.number_calls = 0
        y = func(x)
        assert_equal(y, mock.return_value)
        assert_equal(mock.number_calls, 1)

    assert_allclose(func(x), answer, atol=1e-10)
