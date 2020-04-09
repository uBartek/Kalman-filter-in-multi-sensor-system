def predict(x, p, f, q):
    '''
    these args are not matrices - no transposition

    Parameters:
    x - mean
    p - covariance
    f - transition value
    q - noise

    Returns:
    x - predicted value
    p - predicted covariance
    '''
    x = f * x
    p = f*p*f + q
    return x, p


def update(x, p, h, r, z):
    '''
    these args are not matrices - no transposition

    Parameters:
    x - predicted value
    p - predicted covariance
    h - transition value
    r - measurement noise
    z - measured value 

    Returns:
    x - mean
    p - covariance
    '''
    s = h*p*h + r
    k = p*h/s  # kalman gain
    x = x + k*(z - h*x)
    return x, p
