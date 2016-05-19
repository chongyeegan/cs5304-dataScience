# first line: 1
@mem.cache
def get_data(file_name):
    data = load_svmlight_file(file_name)
    print data[1].shape
    return zip(data[0][:data[0].shape[0]/10,:], np.array(map(lambda x: [0,1] if x == -1 else [1,0], data[1][:data[0].shape[0]/10]))), data[0].shape[1]
