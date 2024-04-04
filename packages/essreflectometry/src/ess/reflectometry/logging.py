# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen
"""
Utilities for logging.

All code in the ess package should log auxiliary information
through this module.
Loggers should be obtained through :py:func:`ess.logging.get_logger`
and pass a name that reflects the current context.
E.g., in the loki package, pass ``'loki'`` (all lowercase).

Logging can be configured using :py:func:`ess.logging.configure` or
:py:func:`ess.logging.configure_workflow`.
Use the latter at the beginning of workflow notebooks.
"""

import functools
import inspect
import logging
import logging.config
from copy import copy
from os import PathLike
from typing import Any, Callable, List, Optional, Sequence, Union

import scipp as sc
import scippneutron as scn
from scipp.utils import running_in_jupyter


def get_logger(subname: Optional[str] = None) -> logging.Logger:
    """Return one of ess's loggers.

    Parameters
    ----------
    subname:
        Name of an instrument, technique, or workflow.
        If given, return the logger with the given name
        as a child of the ess logger.
        Otherwise, return the general ess logger.

    Returns
    -------
    :
        The requested logger.
    """
    name = 'scipp.ess' + ('.' + subname if subname else '')
    return logging.getLogger(name)


def log_call(
    *, instrument: str, message: str = None, level: Union[int, str] = logging.INFO
):
    """
    Decorator that logs a message every time the function is called.
    """
    level = logging.getLevelName(level) if isinstance(level, str) else level

    def deco(f: Callable):
        @functools.wraps(f)
        def impl(*args, **kwargs):
            if message is not None:
                get_logger(instrument).log(level, message)
            else:
                get_logger(instrument).log(level, 'Calling %s', _function_name(f))
            return f(*args, **kwargs)

        return impl

    return deco


class Formatter(logging.Formatter):
    """
    Logging formatter that indents messages and optionally shows threading information.
    """

    def __init__(self, show_thread: bool, show_process: bool):
        """
        Initialize the formatter.

        The formatting is mostly fixed.
        Only printing of thread and processor names can be toggled using the
        corresponding arguments.
        Times are always printed in ISO 8601 format.
        """
        fmt_proc = '%(processName)s' if show_process else ''
        fmt_thread = '%(threadName)s' if show_thread else ''
        if show_process:
            fmt_proc_thread = fmt_proc
            if show_thread:
                fmt_proc_thread += ',' + fmt_thread
        elif show_thread:
            fmt_proc_thread = fmt_thread
        else:
            fmt_proc_thread = ''
        fmt_pre = '[%(asctime)s] %(levelname)-8s '
        fmt_post = '<%(name)s> : %(message)s'
        fmt = (
            fmt_pre
            + ('{' + fmt_proc_thread + '} ' if fmt_proc_thread else '')
            + fmt_post
        )
        super().__init__(fmt, datefmt='%Y-%m-%dT%H:%M:%S%z')

    def format(self, record: logging.LogRecord) -> str:
        record = copy(record)
        record.msg = '\n    ' + record.msg.replace('\n', '\n    ')
        return super().format(record)


def default_loggers_to_configure() -> List[logging.Logger]:
    """
    Return a list of all loggers that get configured by ess by default.
    """
    import pooch

    return [
        sc.get_logger(),
        logging.getLogger('Mantid'),
        pooch.get_logger(),
    ]


def configure(
    *,
    filename: Optional[Union[str, PathLike]] = 'scipp.ess.log',
    file_level: Union[str, int] = logging.INFO,
    stream_level: Union[str, int] = logging.WARNING,
    widget_level: Union[str, int] = logging.INFO,
    show_thread: bool = False,
    show_process: bool = False,
    loggers: Optional[Sequence[Union[str, logging.Logger]]] = None,
):
    """Set up logging for the ess package.

    This function is meant as a helper for application (or notebook) developers.
    It configures the loggers of ess, scippneutron, scipp, and some
    third party packages.
    *Calling it from a library should be avoided*
    because it can mess up a user's setup.

    Up to 3 handlers are configured:

      - *File Handler* Writes files to a file with given name.
        Can be disabled using the `filename` argument.
      - *Stream Handler* Writes to `sys.stderr`.
      - *Widget Handler* Writes to a :py:class:`scipp.logging.LogWidget`
        if Python is running in a Jupyter notebook.

    Parameters
    ----------
    filename:
        Name of the log file. Overwrites existing files.
        Setting this to `None` disables logging to file.
    file_level:
        Log level for the file handler.
    stream_level:
        Log level for the stream handler.
    widget_level:
        Log level for the widget handler.
    show_thread:
        If `True`, log messages include the name of the thread
        the message originates from.
    show_process:
        If `True`, log messages include the name of the process
        the message originates from.
    loggers:
        Collection of loggers or names of loggers to configure.
        If not given, uses :py:func:`default_loggers_to_configure`.

    See Also
    --------
    ess.logging.configure_workflow:
        Configure logging and do some additional setup for a reduction workflow.
    """
    if configure.is_configured:
        get_logger().warning(
            'Called `logging.configure` but logging is already configured'
        )
        return

    handlers = _make_handlers(
        filename, file_level, stream_level, widget_level, show_thread, show_process
    )
    base_level = _base_level([file_level, stream_level, widget_level])
    loggers = {
        logging.getLogger(logger) if isinstance(logger, str) else logger
        for logger in (default_loggers_to_configure() if loggers is None else loggers)
    }
    for logger in loggers:
        _configure_logger(logger, handlers, base_level)
    if any(logger.name == 'Mantid' for logger in loggers):
        _configure_mantid_logging('notice')

    configure.is_configured = True


configure.is_configured = False


def configure_workflow(
    workflow_name: Optional[str] = None, *, display: Optional[bool] = None, **kwargs
) -> logging.Logger:
    """Configure logging for a reduction workflow.

    Configures loggers, logs a greeting message, sets up a logger for a workflow,
    and optionally creates and displays a log widget.

    Parameters
    ----------
    workflow_name:
        Used as the name of the returned logger.
    display:
        If `True`, show a :py:class:`scipp.logging.LogWidget`
        in the outputs of the current cell.
        Defaults to `True` in Jupyter and `False` otherwise.
    kwargs:
        Forwarded to :py:func:`ess.logging.configure`.
        Refer to that function for details.

    Returns
    -------
    :
        A logger for use in the workflow.

    See Also
    --------
    ess.logging.configure:
        General logging setup.
    """
    configure(**kwargs)
    greet()
    if (display is None and running_in_jupyter()) or display:
        sc.display_logs()
    return get_logger(workflow_name)


def greet():
    """Log a message showing the versions of important packages."""
    # Import here so we don't import from a partially built package.
    from . import __version__

    msg = f'''Software Versions:
  ess: {__version__} (https://scipp.github.io/ess)
  scippneutron: {scn.__version__} (https://scipp.github.io/scippneutron)
  scipp: {sc.__version__} (https://scipp.github.io)'''
    mantid_version = _mantid_version()
    if mantid_version:
        msg += f'\n  Mantid: {mantid_version} (https://www.mantidproject.org)'
    get_logger().info(msg)


_INSTRUMENTS = [
    'amor',
    'beer',
    'bifrost',
    'cspec',
    'dream',
    'estia',
    'freia',
    'heimdal',
    'loki',
    'magic',
    'miracles',
    'nmx',
    'odin',
    'skadi',
    'trex',
    'v20',
    'vespa',
]


def _deduce_instrument_name(f: Any) -> Optional[str]:
    # Assumes package name: ess.<instrument>[.subpackage]
    package = inspect.getmodule(f).__package__
    components = package.split('.', 2)
    try:
        if components[0] == 'ess':
            candidate = components[1]
            if candidate in _INSTRUMENTS:
                return candidate
    except IndexError:
        pass
    return None


def _function_name(f: Callable) -> str:
    if hasattr(f, '__module__'):
        return f'{f.__module__}.{f.__name__}'
    return f.__name__


def _make_stream_handler(
    level: Union[str, int], show_thread: bool, show_process: bool
) -> logging.StreamHandler:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(Formatter(show_thread, show_process))
    return handler


def _make_file_handler(
    filename: Union[str, PathLike],
    level: Union[str, int],
    show_thread: bool,
    show_process: bool,
) -> logging.FileHandler:
    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(level)
    handler.setFormatter(Formatter(show_thread, show_process))
    return handler


def _make_handlers(
    filename: Optional[Union[str, PathLike]],
    file_level: Union[str, int],
    stream_level: Union[str, int],
    widget_level: Union[str, int],
    show_thread: bool,
    show_process: bool,
) -> List[logging.Handler]:
    handlers = [_make_stream_handler(stream_level, show_thread, show_process)]
    if filename is not None:
        handlers.append(
            _make_file_handler(filename, file_level, show_thread, show_process)
        )
    if running_in_jupyter():
        handlers.append(sc.logging.make_widget_handler())
    return handlers


def _configure_logger(
    logger: logging.Logger, handlers: List[logging.Handler], level: Union[str, int]
):
    for handler in handlers:
        logger.addHandler(handler)
    logger.setLevel(level)


def _configure_mantid_logging(level: str):
    try:
        from mantid.utils.logging import log_to_python

        log_to_python(level)
    except ImportError:
        pass


def _base_level(levels: List[Union[str, int]]) -> int:
    return min(
        (
            logging.getLevelName(level) if isinstance(level, str) else level
            for level in levels
        )
    )


def _mantid_version() -> Optional[str]:
    try:
        import mantid

        return mantid.__version__
    except ImportError:
        return None
