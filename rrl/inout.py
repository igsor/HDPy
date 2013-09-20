"""
When storing experimental data in HDF5 files, some extra operations
may be useful to process them on a low level. The operations provided
by this module mangle HDF5 files directly (through h5py), without
relying on higher-level functionality. In turn, some of the
functionality may be useful for more advanced stuff.

Note that all functions rely on a specific file format, specifically on
the format which is written by :py:class:`PuPy.RobotCollector`, with
experiments in groups and sensor data in seperate datasets within the
experiment group. On this ground, too short experiments can be removed
(:py:func:`remove_init_only_groups`) or files merged together
(:py:func:`h5_merge_experiments`). When data is split up between two
files, they can easily be put together by :py:class:`H5CombinedFile`.

"""
import h5py
import warnings

def remove_init_only_groups(pth, init_steps):
    """Remove groups from HDF5 data files, which
    
    a) Are empty (0 members)
    b) Have collected less than ``init_steps`` epochs
    
    """
    if isinstance(pth, str):
        f = h5py.File(pth, 'a')
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
    """Rearrange the experiments in ``pth`` such that the experiment
    indices are in the range [0,N], without missing ones.
    No order of the experiments is guaranteed.
    
    """
    if isinstance(pth, str):
        f = h5py.File(pth, 'a')
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

def h5_merge_experiments(pth0, pth1, trg=None):
    """Merge groups of the HDF5 files ``pth0`` and ``pth1``. If ``trg``
    is given, a new file will be created. Otherwise the data is merged
    into ``pth0``.
    
    """
    fh1 = h5py.File(pth1, 'r')
    
    if trg is None:
        f_trg = fh0 = h5py.File(pth0, 'a')
    else:
        f_trg = h5py.File(trg, 'w')
        fh0 = h5py.File(pth0, 'r')
        # Copy groups of file0 to trg
        for k in fh0.keys():
            fh0.copy(k, f_trg)
    
    groups_0 = map(int, fh0.keys())
    groups_1 = map(int, fh1.keys())
    
    # Copy groups of file1 to trg
    offset = 1 + max(groups_0) - min(groups_1)
    for k in groups_1:
        src = str(k)
        dst = str(k + offset)
        fh1.copy(src, f_trg, name=dst)
    
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
        f = h5py.File(pth, 'a')
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

class H5CombinedFile(object):
    """Combine two HDF5 files which have the same groups on the root
    level but different datasets within these groups. The files are
    packed together such that they can be handled as if a single file
    was present. 
    
    ``pth_main``
        Path to the first HDF5 file. If a dataset is available in
        both files, the one from this file will be used.
        
    
    ``pth_additional``
        Path to the second HDF5 file.
    
    """
    def __init__(self, pth_main, pth_additional):
        self.pth0 = pth_main
        self.pth1 = pth_additional
        self.fh0 = h5py.File(pth_main, 'r')
        self.fh1 = h5py.File(pth_additional, 'r')
        self.keys0 = [k for k in self.fh0 if len(self.fh0[k]) > 0]
        self.keys1 = [k for k in self.fh1 if len(self.fh1[k]) > 0]
        self.keys_common = [k for k in self.keys0 if k in self.keys1]
    
    def __getitem__(self, key):
        """Return a :py:class:`DataMergeGroup` instance, binding the
        groups ``key`` of the two files together.
        """
        if key not in self.keys_common:
            raise KeyError()
        
        return H5CombinedGroup(self.fh0[key], self.fh1[key])

    def __len__(self):
        """Return the length of all (shared) groups."""
        return len(self.keys_common)
    
    def __contains__(self, item):
        """True iff ``item`` is a group known in both files."""
        return item in self.keys_common
    
    def keys(self):
        """Return all group names which are present in both files."""
        return self.keys_common[:]
    
    def close(self):
        """Close all filehandlers."""
        self.fh0.close()
        self.fh1.close()
    
    def attributes(self, key):
        """Return two attribute manager instances, one pointing to group
        ``key`` in each file.
        """
        assert key in self.keys_common
        attrs0 = h5py.AttributeManager(self.fh0[key])
        attrs1 = h5py.AttributeManager(self.fh1[key])
        return attrs0, attrs1

class H5CombinedGroup(object):
    """Combine two related HDF5 groups which store different datasets
    and present them as a single group. Instances to this class are
    typically exclusively created through :py:class:`H5CombinedFile`.
    
    ``grp0``
        Group of the first file. If a dataset is present in both groups,
        the one from this group will be used.
    
    ``grp1``
        Group of the second file.
    
    """
    def __init__(self, grp0, grp1):
        self.grp0 = grp0
        self.grp1 = grp1
    
    def __getitem__(self, key):
        """Return dataset ``key`` or raise an exception if neither of
        the groups contains this key.
        """
        if key in self.grp0:
            return self.grp0[key]
        elif key in self.grp1:
            return self.grp1[key]
        else:
            raise KeyError()
    
    def __len__(self):
        """Return the number of keys in both groups."""
        return len(self.grp0) + len(self.grp1)
    
    def __contains__(self, item):
        """True iff ``item`` is a key of one of the groups."""
        return item in self.grp0 or item in self.grp1
    
    def keys(self):
        """Return a list of datasets names found in any of the groups."""
        return self.grp0.keys() + self.grp1.keys()
    
    def attributes(self):
        """Return two attribute manager instances, one pointing to each
        group."""
        attrs0 = h5py.AttributeManager(self.grp0)
        attrs1 = h5py.AttributeManager(self.grp1)
        return attrs0, attrs1


class DataMerge(H5CombinedFile):
    """Identical to :py:class:`H5Combine`
    
    .. deprecated:: 1.0
        Use :py:class:`H5Combine`.
    
    """
    def __init__(self, *args, **kwargs):
        warnings.warn('This class is depcreated. Use H5CombinedFile instead')
        super(DataMerge, self).__init__(*args, **kwargs)

class DataMergeGroup(H5CombinedGroup):
    """Identical to :py:class:`H5CombinedGroup`
    
    .. deprecated:: 1.0
        Use :py:class:`H5CombinedGroup`
    
    """
    def __init__(self, *args, **kwargs):
        warnings.warn('This class is depcreated. Use H5CombinedGroup instead')
        super(DataMergeGroup, self).__init__(*args, **kwargs)

