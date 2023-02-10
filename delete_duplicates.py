more_then = 2


def remove_duplicates(s):
    s = s.split()
    output = []
    i = 0
    while i < len(s):
        orig_world = s[i]
        j = i + 1
        while j < len(s) and s[j] == orig_world:
            j += 1
        if j - i < more_then:
            output.append(orig_world)
        i += 1
    return ' '.join(word for word in output)
