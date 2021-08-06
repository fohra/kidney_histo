import pandas as pd

def sample_infos(infos, num_cancer, num_benign, seed, include_edge = False, include_center=True):
    '''Sample images based on the desired ratio
    infos (pandas DataFrame):
        DataFrame containing image paths and infos
    num_cancer (int):
        How many cancer images is sampled 
    num_benign (int):
        How many benign images is sampled 
    include_edge (bool):
        Whether to include edge spots as cancer
    include_center (bool):
        Whether to include center spots as cancer
        
    Returns:
        cancer paths
        benign paths
        DataFrame
    '''
    # Separate cancer and benign infos
    if include_edge & include_center:
        cancer = infos[(infos['Annotation']=='Center') | (infos['Annotation']=='Edge')].copy()
    elif include_edge:
        cancer = infos[infos['Annotation']=='Edge'].copy()
    else:
        cancer = infos[infos['Annotation']=='Center'].copy()
    benign = infos[infos['Annotation']=='Normal'].copy()
    
    #if both too large
    if num_cancer > len(cancer) & num_benign > len(benign):
        raise Exception('Not enough images in cancer and benign sets. Number of benign images: '+str(len(benign))+
                        '. Number of cancer images: '+str(len(cancer)))
        
    #if one too large
    elif num_cancer > len(cancer):
        raise Exception('Not enough images in cancer set. Number of cancer images: '+str(len(cancer)))
        
    #if other too large
    elif num_benign > len(benign):
        raise Exception('Not enough images in benign set. Number of benign images: '+str(len(benign)))
    
    #suffle
    if not num_cancer == len(cancer):
        cancer = cancer.sample(n=num_cancer, random_state=seed).reset_index(drop=True)
    if not num_benign == len(benign):
        benign = benign.sample(n=num_benign, random_state=seed).reset_index(drop=True)
    ret_df = pd.concat([cancer,benign], ignore_index=True)
    
    return ret_df