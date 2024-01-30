def load(options, key):
    (key,) = set(filter(lambda x: x.startswith(key), options.keys()))
    fn, args, kwargs = options[key]
    return fn(*args, **kwargs)
