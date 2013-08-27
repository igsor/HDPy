
import h5py

def remove_init_only_groups(pth, init_steps):
    """Remove groups from HDF5 data files, which
    
    a) Are empty (0 members)
    b) Have collected less than ``init_steps`` epochs
    
    """
    if isinstance(pth, str):
        f = h5py.File(pth,'a')
    else:
        f = pth
    
    all_keys = f.keys()
    remove_zero = [k for k in all_keys if len(f[k]) == 0]
    remove_short = [k for k in all_keys if len(f[k]) > 0 and f[k]['a_curr'].shape[0] < init_steps]
    
    for k in remove_zero + remove_short:
        print "Removing", k
        del f[k]
    
    print "Removed", (len(remove_zero) + len(remove_short)), "groups"
    
    return f

def h5_reorder(pth):
    """
    Rearrange the experiments in ``pth`` such that the experiment
    indices are in the range [0,N], without missing ones.
    No order of the experiments is guaranteed.
    
    """
    if isinstance(pth, str):
        f = h5py.File(pth,'a')
    else:
        f = pth
    
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
    
    return f

def h5_merge_datasets(pth0, pth1, trg=None):
    """
    
    .. todo::
        not implemented, undocumented, untested
    
    """
    raise NotImplementedError()

def h5_merge_experiments(pth0, pth1, trg=None):
    """Merge groups of the HDF5 files ``pth0`` and ``pth1``. If ``trg``
    is given, a new file will be created. Otherwise the data is merged
    into ``pth0``.
    
    """
    f1 = h5py.File(pth1, 'r')
    
    if trg is None:
        f_trg = f0 = h5py.File(pth0, 'a')
        
    else:
        f_trg = h5py.File(trg, 'w')
        f0 = h5py.File(pth0, 'r')
        # Copy groups of file0 to trg
        for k in f0.keys():
            f0.copy(k, f_trg)
    
    groups_0 = map(int, f0.keys())
    groups_1 = map(int, f1.keys())
    
    # Copy groups of file1 to trg
    offset = 1 + max(groups_0) - min(groups_1)
    for k in groups_1:
        src = str(k)
        dst = str(k + offset)
        f1.copy(src, f_trg, name=dst)
    
    return f_trg

def remove_boundary_groups(pth):
    """Remove the first and last experiment with respect to webots
    restart/revert in ``pth``. The boundaries are determined through
    the *init_step* group. This method is to save possibly corrupted
    experimental data files, due to webots' memory issues. To work
    properly, the groups must not be altered before this method, e.g.
    by :py:func:`remove_init_only_groups`.
    """
    if isinstance(pth, str):
        f = h5py.File(pth,'a')
    else:
        f = pth
    
    keys = sorted(map(int, f.keys()))
    restarts = [k for k in keys if 'init_step' in f[str(k)]]
    restarts += [k-1 for k in restarts if k > 0]
    restarts += [keys[-1]]
    restarts = set(sorted(restarts))
    for k in restarts:
        del f[str(k)]
    
    return f

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
    
    def attributes(self, key):
        assert key in self.keys_common
        attrs0 = h5py.AttributeManager(self.f0[key])
        attrs1 = h5py.AttributeManager(self.f1[key])
        return attrs0, attrs1

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
    
    def attributes(self):
        attrs0 = h5py.AttributeManager(self.grp0)
        attrs1 = h5py.AttributeManager(self.grp1)
        return attrs0, attrs1
