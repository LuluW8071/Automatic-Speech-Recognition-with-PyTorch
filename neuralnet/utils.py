import torch

class TextTransform:
    """ Maps characters to integers and vice versa """
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '


    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        special_chars = {' ', '–', '.', '-'}     # Some special chars 
        for c in text:
            if c in special_chars:
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
                # print('Mapped Chars:', ch)
            int_sequence.append(ch)
        return int_sequence


    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to a text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i]) 
            # print(string[0])
        return ''.join(string).replace('<SPACE>', ' ')
    
# Initialize TextProcess for text processing
text_transform = TextTransform()

def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    # print(output)
    arg_maxes = torch.argmax(output, dim=2)
    # print('Argmax indices: ', arg_maxes[0]) 
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j-1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets