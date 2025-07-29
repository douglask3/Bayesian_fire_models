import os

def makeDir(path):
    if not os.path.exists(path): os.makedirs(path)

def combine_path_and_make_dir(path1, path2 = ""):
    """pastes two paths together and makes that directory if it doesnt already exist   
    Arguments:
        path1 -- string - directory path
	path2 --string - directory path or blank (defualt is to blank)
    Returns:
        path of resultant directory
    """
    path = path1 + '/'+ path2 + '/'
    makeDir(path)
    return path
