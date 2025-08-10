def flatten(xss):
    # Purpose:
    #   Flattens a list of lists into a single list.
    #
    # Input:
    #   xss - A list of lists (e.g., [[1, 2], [3], [4, 5]])
    #
    # Output:
    #   A flat list containing all elements from the sublists
    #   (e.g., [1, 2, 3, 4, 5])
    #
    # Example:
    #   flatten([[1, 2], [3], [4, 5]])  -->  [1, 2, 3, 4, 5]
    return [x for xs in xss for x in xs]

