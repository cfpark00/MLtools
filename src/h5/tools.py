import h5py

def repack(h5_file_path):
    """
    Repack the h5 file to reset disk usage.

    h5_file_path: str
        The path to the h5 file.

    """
    h5=h5py.File(h5_file_path,"r")
    h5new=h5py.File(h5_file_path+"_temp","w")
    for key,val in h5.items():
        h5.copy(key,h5new)
    for key,val in h5.attrs.items():
        h5new.attrs[key]=val
    h5.close()
    h5new.close()
    os.remove(h5_file_path)
    os.rename(h5_file_path+"_temp",h5_file_path) 
