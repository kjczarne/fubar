def filepattern(pattern, extension, defaulttag, analysistype=""):
    """
    generates pattern names for efficient exporting of files, great for iterative saving of model parameters as HDF5
    and architechtures as JSON when working with keras

    example call: filepattern('hist_ana_', '.pkl', '5.0', 'convolution_stack) -> hist_ana_5.0convolution_stack.pkl
    above is true provided there is no version tag in the directory higher than 5.0

    :param pattern: defines starting pattern of a file
    :param extension: defines searched file extension
    :param defaulttag: defines default tag if no file that matches pattern is found
    :param analysistype: additional tag for analysis file naming
    :return: returns a filename that follows the same pattern but has higher tag by 0.1
    """
    lst = []
    expression = pattern + '[0-9].[0-9]*' + extension
    # above matches two digit tag separated by dot and accepts any number of additional
    # characters before extension is matched

    for i in glob.glob(expression):
        i = i[(len(pattern)):(len(pattern) + 3)]
        i = Decimal(i)
        lst.append(i.quantize(Decimal('0.1'), rounding='ROUND_DOWN'))
    # finds patterns and saves tags in a list as Decimal objects

    if len(lst) != 0:
        newtag = Decimal('0.1') + Decimal(max(lst))
        newtag = str(newtag)
    else:
        pass  # do nothing (last conditional solves this case already)

    if defaulttag is None:
        defaulttag = Decimal('0.0')
    else:
        defaulttag = Decimal(defaulttag) + Decimal('0.1')

    if len(lst) == 0:
        filename = pattern + str(defaulttag) + analysistype + extension
    else:
        filename = pattern + newtag + analysistype + extension
    return filename