from sklearn.model_selection import KFold

def get_idx_label(seed,sub=0):
    
    
    label_idx = [i for i in range(2700*7)]
    if sub==0:
        kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    else:
        kfold = KFold(n_splits=4, shuffle=True,random_state=seed)
    train_ids, valid_ids = list(), list()
    
    for idx, (train_index, valid_index) in enumerate(kfold.split(label_idx)):
        train_ids.append(train_index)
        valid_ids.append(valid_index)
    
    # train data index와 valid data index
    return train_ids, valid_ids


def get_idx_people(seed):
    '''
    KFold, Split indexes by people.
    '''
    label_idx = [i for i in range(2700)]
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    train_num, valid_num = list(), list()
    
    for idx, (train_idx, valid_idx) in enumerate(kfold.split(label_idx)):
        train_idx = train_idx * 7
        valid_idx = valid_idx * 7
        train_tmp, valid_tmp = list(), list()
        
        for i in train_idx:
            for j in range(7):
                train_tmp.append(i + j)
        
        for i in valid_idx:
            for j in range(7):
                valid_tmp.append(i + j)

        train_num.append(train_tmp)
        valid_num.append(valid_tmp)
    return train_num, valid_num