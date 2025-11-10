import os

def hasNewFilesRecent(watchDir, knownFiles=None):
    """
    Check if new files are present in a directory.
    ONLY returns the name of the most recent file, even if multiple have been added.

    Args:
        watchDir (str): Path to the directory to monitor.
        knownFiles (set[str] | None): Previously seen files. If None, initializes from current files.

    Returns:
        (bool, set[str], str | None):
            - True if new files found
            - Updated file set
            - Full path of the new file, or None if none found
    """
    currentFiles = set(os.listdir(watchDir))

    if knownFiles is None:
        # First run: initialize baseline without reporting new files
        return False, currentFiles, None

    newFiles = currentFiles - knownFiles
    if newFiles:
        # Pick the one with the most recent modification time
        newFile = max(
            newFiles,
            key=lambda f: os.path.getmtime(os.path.join(watchDir, f))
        )
        newFilePath = os.path.join(watchDir, newFile)
        return True, currentFiles, newFilePath
    else:
        return False, currentFiles, None


    import os

def hasNewFiles(watchDir, knownFiles=None):
    """
    Check if new files are present in a directory.

    Args:
        watchDir (str): Path to the directory to monitor.
        knownFiles (set[str] | None): Previously seen files. If None, initializes from current files.

    Returns:
        (bool, set[str], list[str] | None):
            - True if new files found
            - Updated file set
            - List of full paths of new files, or None if none found
    """
    currentFiles = set(os.listdir(watchDir))

    if knownFiles is None:
        # First run: initialize baseline without reporting new files
        return False, currentFiles, None

    newFiles = currentFiles - knownFiles
    if newFiles:
        # Return all new files with full paths
        newFilePaths = [os.path.join(watchDir, f) for f in newFiles]
        return True, currentFiles, newFilePaths
    else:
        return False, currentFiles, None

