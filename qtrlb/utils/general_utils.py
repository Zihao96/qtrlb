def make_it_list(thing, default: list = None):
    """
    A crucial, life-saving function.
    """
    if isinstance(thing, list):
        return thing
    elif thing is None:
        return [] if default is None else default
    else:
        return [thing]