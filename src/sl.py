# functions taken from: https://github.com/ocropus/ocropy/blob/master/ocrolib/sl.py

################################################################
# utilities for lists of slices, treating them like rectangles
################################################################


def start(u):
    return tuple([x.start for x in u])


def raster(u):
    """Return (col0,row0,col1,row1)."""
    return (u[1].start, u[0].start, u[1].stop, u[0].stop)


def box(c0, r0, c1, r1):
    return (slice(r0, r1), slice(c0, c1))


def shift(u, offsets):
    u = list(u)
    for i in range(len(offsets)):
        u[i] = slice(u[i].start+offsets[i], u[i].stop+offsets[i])
    return tuple(u)


def pad(u, d):
    """Pad the slice list by the given amount."""
    return tuple([slice(u[i].start-d, u[i].stop+d) for i in range(len(u))])


def union(u, v):
    """Compute the union of the two slice lists."""
    if u is None:
        return v
    if v is None:
        return u
    return tuple([slice(min(u[i].start, v[i].start), max(u[i].stop, v[i].stop)) for i in range(len(u))])


def width(s):
    return s[1].stop-s[1].start


def height(s):
    return s[0].stop-s[0].start
