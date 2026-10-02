def __getattr__(name):
    if name == 'G1Adapter':
        from .adapter import G1Adapter
        return G1Adapter
    raise AttributeError(name)
