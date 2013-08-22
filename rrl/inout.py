
import h5py

def h5_merge(pth0, pth1):
    """
    
    .. todo::
        undocumented, untested
    
    """
    raise NotImplementedError()

class DataMerge:
    """
    
    .. todo::
        undocumented, untested
    
    """
    def __init__(self, pth_main, pth_additional):
        self.pth0 = pth_main
        self.pth1 = pth_additional
        self.f0 = h5py.File(pth_main, 'r')
        self.f1 = h5py.File(pth_additional, 'r')
        assert len(self.f0) == len(self.f1)
    
    def __getitem__(self, key):
        return DataMergeGroup(self.f0[key], self.f1[key])

    def __len__(self):
        return len(self.f0)
    
    def __contains__(self, item):
        return item in self.f0 and item in self.f1
    
    def keys(self):
        k0 = self.f0.keys()
        k1 = self.f1.keys()
        return [k for k in k0 if k in k1]
    
    def close(self):
        self.f0.close()
        self.f1.close()

class DataMergeGroup:
    """
    
    .. todo::
        undocumented, untested
    
    """
    def __init__(self, grp0, grp1):
        self.grp0 = grp0
        self.grp1 = grp1
    
    def __getitem__(self, key):
        if key in self.grp0:
            return self.grp0[key]
        elif key in self.grp1:
            return self.grp1[key]
        else:
            raise Exception('')
    
    def __len__(self):
        return len(self.grp0) + len(self.grp1)
    
    def __contains__(self, item):
        return item in self.grp0 or item in self.grp1
    
    def keys(self):
        return self.grp0.keys() + self.grp1.keys()
    
