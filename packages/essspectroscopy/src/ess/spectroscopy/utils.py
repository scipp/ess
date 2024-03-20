from scipp import Variable


def norm(vector: Variable) -> Variable:
    from scipp import sqrt, dot, DType
    assert vector.dtype == DType.vector3  # "Vector operations require scipp.DType.vector3 elements!"
    return sqrt(dot(vector, vector))

