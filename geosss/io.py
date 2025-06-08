import errno
import os
import pickle as Pickle
import time


def dump(this, filename, gzip=False, lock=None, timeout=None):
    """
    Write a Python object to a file using pickle.

    Supports tilde expansion for home directory paths ('~' or '~user').

    Args:
        this: The object to be written to disk.
        filename (str): Path to the output file.
        gzip (bool, optional): If True, compress the file with gzip. Defaults to False.
        lock (bool, optional): If True, use a lockfile to restrict access. Defaults to None.
        timeout (float, optional): Timeout in seconds for lock acquisition. Defaults to None.

    Raises:
        IOError: If lock acquisition fails or lockfile operations fail.
    """
    filename = os.path.expanduser(filename)

    # Acquire lock if requested
    if lock is not None:
        lockdir = filename + ".lock"

        if timeout is not None and timeout > 0:
            end_time = timeout + time.time()

        while True:
            try:
                os.mkdir(lockdir)
                break
            except OSError as e:
                if e.errno == errno.EEXIST:
                    if timeout is not None and time.time() > end_time:
                        raise IOError("Failed to acquire Lock")
                    time.sleep(0.01)  # Small delay to prevent busy waiting
                else:
                    raise IOError("Failed to acquire Lock")

    # Open file stream
    if gzip:
        import gzip

        stream = gzip.GzipFile(filename, "wb")
    else:
        stream = open(filename, "wb")

    try:
        Pickle.dump(this, stream, protocol=2)
    finally:
        stream.close()

        # Release lock if acquired
        if lock is not None:
            try:
                os.rmdir(lockdir)
            except OSError:
                raise IOError("missing lockfile {0}".format(lockdir))


def load(filename, gzip=False, lock=None, timeout=None):
    """
    Load a Python object from a pickled file.

    Supports tilde expansion for home directory paths ('~' or '~user').

    Args:
        filename (str): Path to the pickled file.
        gzip (bool, optional): If True, decompress the file with gzip. Defaults to False.
        lock (bool, optional): If True, use a lockfile to restrict access. Defaults to None.
        timeout (float, optional): Timeout in seconds for lock acquisition. Defaults to None.

    Returns:
        The unpickled Python object.

    Raises:
        IOError: If lock acquisition fails, file cannot be read, or unpickling fails.
    """
    filename = os.path.expanduser(filename)

    # Acquire lock if requested
    if lock is not None:
        lockdir = filename + ".lock"

        if timeout is not None and timeout > 0:
            end_time = timeout + time.time()

        while True:
            try:
                os.mkdir(lockdir)
                break
            except OSError as e:
                if e.errno == errno.EEXIST:
                    if timeout is not None and time.time() > end_time:
                        raise IOError("Failed to acquire Lock")
                    time.sleep(0.01)  # Small delay to prevent busy waiting
                else:
                    raise IOError("Failed to acquire Lock")

    # Determine file type and open stream
    if gzip:
        import gzip

        stream = gzip.GzipFile(filename, "rb")
        try:
            # Test if gzip file is readable
            stream.readline()
            stream.seek(0)
        except (OSError, IOError) as e:
            stream.close()
            raise IOError("Cannot read gzip file: {0}".format(e))
    else:
        stream = open(filename, "rb")

    try:
        this = Pickle.load(stream)
    except (Pickle.UnpicklingError, EOFError, IOError) as e:
        raise IOError("Failed to unpickle file: {0}".format(e))
    finally:
        stream.close()

        # Release lock if acquired
        if lock is not None:
            try:
                os.rmdir(lockdir)
            except OSError:
                raise IOError("missing lockfile {0}".format(lockdir))

    return this
