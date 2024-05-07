    """Masker for training a JEPA model
    
    To do this we will need to mask a random (within a specified range) portion of the input data. Than createa 
    a series of target blocks for the decoder to predict.
    
    
    we take the original input and mask a portion of this, this is the "context"
    
    We than take the original input and mask most of it only leaving a part not included in the context
    this is the "target"
    We do this 4 times for each image that is passed in than these targets are used as the training objectives 
    for the decoder.
    
    """