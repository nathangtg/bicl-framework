def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()