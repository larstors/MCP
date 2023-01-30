import numpy as np

# what we want
a = 9588514242


def flip_nr(b: int):
    """Find number of spins to flip

    Args:
        b (int): what we have
    """
    c = b
    d = b
    length = b.bit_length()

    c = c >> b.bit_length() - 1
    d = d % (2 ** (b.bit_length() - 1))
    length = length - d.bit_length()
    d = d << length
    c = c << length - 1
    check = length
    d += c
    c = d
    print(bin(d), bin(b), d.bit_length(), b.bit_length())

    while d != b:
        length = b.bit_length()

        print(bin(d), bin(b), d.bit_length(), b.bit_length())

        c = c >> b.bit_length() - 1
        d = d % (2 ** (b.bit_length() - 1))
        length = length - d.bit_length()
        d = d << length
        c = c << length - 1
        check += length
        d += c
        c = d

    return check


print("?????", flip_nr(a))

print(bin(3), bin(3 << 1), bin(3 >> 1))
print(int("1000111011100001010001110111000010", 2))
