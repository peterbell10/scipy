import scipy.fft as fft
from . import _config as cfg
from contextlib import contextmanager

class BackendError(RuntimeError):
    pass

class BackendWarning(UserWarning):
    pass

def _wrap_fallback(fname, on_missing):
    """Wrap fallback function with error reporting according to on_missing"""
    message = "The current backend does not implement '{}'".format(fname)

    if on_missing == 'fallback':
        return cfg._fallbacks[fname]
    elif on_missing == 'warn':
        def wrap_warn(*args, **kwargs):
            from warnings import warn
            warn(message, BackendWarning)
            return cfg._fallbacks[fname](*args, **kwargs)
        return wrap_warn
    elif on_missing == 'raise':
        def wrap_raise(*args, **kwargs):
            raise BackendError(message)
        return wrap_raise

    raise ValueError("Unrecognized on_missing type '{}'".format(on_missing))



def set_backend(backend, on_missing='fallback'):
    """Sets the current fft backend

    Parameters
    ----------

    backend: string
        Can either be one of the known backends {'scipy'}, or a
        module import specification of the form 'module://example.fft'
    on_missing: {'fallback', 'warn', 'raise'}, optional
        Behavior when the backend does not provide a given function:
        - 'fallback': silently use the built-in SciPy function
        - 'warn': emit a warning, then use SciPy's default
        - 'raise': raise an error

    Raises
    ------
    ImportError: If the specified backend could not be imported
    ValueError: If an invalid parameter is given

    """

    # Reset all to default
    cfg._functions = cfg._fallbacks.copy()

    if backend == 'scipy':
        return

    if backend.startswith('module://'):
        import importlib
        backend_module = importlib.import_module(backend[9:])
    else:
        raise ValueError('Unrecognized backend {}'.format(backend))

    for fname in cfg._functions.keys():
        fallback = _wrap_fallback(fname, on_missing)
        cfg._functions[fname] = getattr(backend_module, fname, fallback)


@contextmanager
def backend(backend, on_missing='fallback'):
    """Context manager to change the backend within a fixed scope

    Upon entering a ``with`` statement, the current backend is changed. Upon
    exit, the backend is reset to the state before entering the scope.

    Parameters
    ---------
    backend: string
        Can either be one of the known backends {'scipy'}, or a
        module import specification of the form 'module://example.fft'
    on_missing: {'fallback', 'warn', 'raise'}, optional
        Behavior when the backend does not provide a given function:

    Examples
    --------
    >>> with scipy.fft.backend('scipy'):
    >>>     pass

    """
    old_functions = cfg._functions.copy()
    set_backend(backend, on_missing)
    yield
    cfg._functions = old_functions
