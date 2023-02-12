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

def no_dups(s, thres=3):
    for num_of_words in range(1, int(len(s) / thres)):
        s = s.split()
        i = 0
        while i + num_of_words < len(s):
            orig_seq = s[i:i + num_of_words]
            j = i + num_of_words
            dif_seq = s[j: j + num_of_words]
            while j + num_of_words < len(s) and dif_seq == orig_seq:
                j += num_of_words
                dif_seq = s[j:j + num_of_words]
            if j - i > thres * num_of_words:
                return False
            i += 1
    return True
