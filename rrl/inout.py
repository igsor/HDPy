
import h5py

def remove_init_only_groups(pth, init_steps, reorder=False):
    """Remove groups from HDF5 data files, which
    
    a) Are empty (0 members)
    b) Have collected less than ``init_steps`` epochs
    
    If ``reorder``, the experiments are rearranged such that the
    experiment indices are in the range [0,N], without missing ones.
    No order of the experiments is guaranteed.
    
    """
    f = h5py.File(pth,'a')
    all_keys = f.keys()
    remove_zero = [k for k in all_keys if len(f[k]) == 0]
    remove_short = [k for k in all_keys if len(f[k]) > 0 and f[k]['a_curr'].shape[0] < init_steps]
    
    for k in remove_zero + remove_short:
        print "Removing", k
        del f[k]
    
    print "Removed", (len(remove_zero) + len(remove_short)), "groups"
    
    if reorder:
        # keys must be ascending
        old_keys = map(str, sorted(map(int, f.keys())))
        for new_key, old_key in enumerate(old_keys):
            new_key = str(new_key)
            if new_key != old_key:
                if new_key not in f.keys():
                    print old_key, "->", new_key
                    f[new_key] = f[old_key]
                    del f[old_key]
                else:
                    print "Cannot move", old_key, "to", new_key, "(new key exists)"
        
    
    f.close()

def h5_merge(pth0, pth1):
    """
    
    .. todo::
        not implemented, undocumented, untested
    
    """
    raise NotImplementedError()

class DataMerge:
    """
    
    .. todo::
        undocumented
    
    """
    def __init__(self, pth_main, pth_additional):
        self.pth0 = pth_main
        self.pth1 = pth_additional
        self.f0 = h5py.File(pth_main, 'r')
        self.f1 = h5py.File(pth_additional, 'r')
        self.keys0 = [k for k in self.f0 if len(self.f0[k]) > 0]
        self.keys1 = [k for k in self.f1 if len(self.f1[k]) > 0]
        self.keys_common = [k for k in self.keys0 if k in self.keys1]
    
    def __getitem__(self, key):
        if key not in self.keys_common:
            raise KeyError()
        
        return DataMergeGroup(self.f0[key], self.f1[key])

    def __len__(self):
        return len(self.keys_common)
    
    def __contains__(self, item):
        return item in self.keys_common
    
    def keys(self):
        return self.keys_common[:]
    
    def close(self):
        self.f0.close()
        self.f1.close()

class DataMergeGroup:
    """
    
    .. todo::
        undocumented
    
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
            raise KeyError()
    
    def __len__(self):
        return len(self.grp0) + len(self.grp1)
    
    def __contains__(self, item):
        return item in self.grp0 or item in self.grp1
    
    def keys(self):
        return self.grp0.keys() + self.grp1.keys()
    
