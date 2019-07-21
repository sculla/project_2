#!/anaconda3/envs/metis/bin/python

"""
unpacking the pickles from the scraper to make one file
"""

if __name__ == '__main__':
    import pickle
    import pandas as pd

    with open('pickle/columns.pickle', 'rb') as f:
        column_list = pickle.load(f)
        f.close()
    with open('pickle/pickle_list.pickle', 'rb') as f:
        name_list = pickle.load(f)
        f.close()

    homes_df = pd.DataFrame(columns=column_list)

    for name in name_list:
        print(name, 'next')
        new_df = pd.read_pickle('pickle/' + name)
        if new_df.shape == (0, 0):
            continue

        new_df.set_axis(column_list, axis=1, inplace=True)

        homes_df = homes_df.append(new_df)
        print(name, 'completed')
        print(homes_df.shape)
    print(homes_df.shape)

    homes_df.to_pickle('data/full_list.pickle')
    print('wheeeeeeeeeee its in!')
