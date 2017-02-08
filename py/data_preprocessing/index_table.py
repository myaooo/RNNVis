def make_table(data, delimiter='\n', gram_num=2):
    '''Construct the index table of the text data
    '''
    if gram_num < 1:
        raise ValueError('The length of n-gram should at least 1')
    if not isinstance(data, str):
        raise ValueError('The data should be str type')
    index_table = {}
    lines = data.split(delimiter)
    for line in lines:
        words = line.split(' ')
        for i in range(len(words) - gram_num + 1):
            gram = ' '.join(words[i:i+gram_num])
            index_table.setdefault(gram, set()).add(i)
    return index_table

if __name__ == '__main__':
    data_path = '/tmp/data/ptb.test.txt'
    with open(data_path, 'r') as file:
        data = file.read()
    index_table = make_table(data)
    print(index_table)





