import re



def find_nth_occurrence(string: str, substring: str, n: int) -> re.Match:
    """
    Return a Match object of the n-th occurrence of substring in string.
    n counts from 0.
    Raise ValueError if n-th occurrence does not exist.
    Example:
        find_nth_occurrence('hello world', 'l', 2)
        Return <re.Match object; span=(9, 10), match='l'>
    """
    for i, rematch in enumerate(re.finditer(substring, string)):
        if i == n: return rematch

    raise ValueError(f'No {n}th occurrence of {substring} in {string}.')


def replace_except_nth_occurrence(string: str, substring: str, new_substring: str, n: int) -> str:
    """
    Replace all occurrences of substring in string except the n-th one.
    n counts from 0.
    Example:
        replace_except_nth_occurrence('hello world', 'l', 'x', 2)
        Return 'hexxo world'
    """
    rematch = find_nth_occurrence(string, substring, n)
    new_string = (
        string[:rematch.start()].replace(substring, new_substring) 
        + string[rematch.start(): rematch.end()]
        + string[rematch.end():].replace(substring, new_substring)
    )
    return new_string


def remove_identical_neighbor_pattern(string: str, pattern: str) -> str:
    """
    Detect certain pattern in regular expression in given string, \
    and remove the repeated identical neighbor substring.
    The original position will be occupied by equal amount of whitespace.
    Example:
        string = '''
                    set_awg_gain     15050,-445
                    play             0,1,40 
        
                    set_awg_gain     7525,-223
                    play             0,1,40 

                    set_awg_gain     7525,-223
                    play             0,1,40 
        
                    set_awg_gain     7525,-223
                    play             0,1,40 
        
                    set_awg_gain     15050,-445
                    play             0,1,40 
        '''
        pattern = '[ \t]+set_awg_gain.*\n'

        return '''
                    set_awg_gain     15050,-445
                    play             0,1,40 
        
                    set_awg_gain     7525,-223
                    play             0,1,40 

                                              
                    play             0,1,40 
        
                                              
                    play             0,1,40 
        
                    set_awg_gain     15050,-445
                    play             0,1,40 
        '''
    """
    old_string = ''

    for rematch in re.finditer(pattern, string):
        new_string = rematch.group()

        if new_string == old_string:
            string = (
                string[:rematch.start()] 
                + ' ' * (len(new_string)-1) + '\n' 
                + string[rematch.end():]
            )
        else:
            old_string = new_string

    return string

