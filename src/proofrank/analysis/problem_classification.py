def fix_class(cls: str) -> str:
    if "geo" in cls or "Geo" in cls:
        return "Geometry"
    elif "umber" in cls:
        return "Number Theory"
    elif "algebra" in cls or "Algebra" in cls:
        return "Algebra"
    elif "combin" in cls or "Combin" in cls:
        return "Combinatorics"
    elif "calc" in cls or "Calc" in cls:
        return "Calculus"
    else:
        return "Unknown"
