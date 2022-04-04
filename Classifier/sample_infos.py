import pandas as pd

def sample_infos(infos, num_cancer, num_benign, seed, num_relapse=0, num_non_relapse=0, num_death = 0, num_alive = 0, include_edge = False, include_center=True):
    '''Sample images based on the desired ratio
    infos (pandas DataFrame):
        DataFrame containing image paths and infos
    num_cancer (int):
        How many cancer images is sampled 
    num_benign (int):
        How many benign images is sampled 
    num_relapse (int):
        How many relapse images is sampled 
    num_non_relapse (int):
        How many non_relapse images is sampled 
    include_edge (bool):
        Whether to include edge spots as cancer
    include_center (bool):
        Whether to include center spots as cancer
        
    Returns:
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
    
    #Raise error if not enough images in dataframes
    if (num_cancer > len(cancer)) & (num_benign > len(benign)):
        raise Exception('Not enough images in cancer and benign sets. Number of benign images: '+str(len(benign))+
                        '. Number of cancer images: '+str(len(cancer)))
    
    elif num_cancer > len(cancer):
        raise Exception('Not enough images in cancer set. Number of cancer images: '+str(len(cancer)))
    
    elif num_benign > len(benign):
        raise Exception('Not enough images in benign set. Number of benign images: '+str(len(benign)))
    
    #sample wanted number of images
    if not num_cancer == len(cancer):
        cancer = cancer.sample(n=num_cancer, random_state=seed).reset_index(drop=True)
    if not num_benign == len(benign):
        benign = benign.sample(n=num_benign, random_state=seed).reset_index(drop=True)
    ret_df = pd.concat([cancer,benign], ignore_index=True)
    
    # Sample relapse if needed
    if (num_relapse + num_non_relapse)> 0:
        relapse = ret_df[ret_df['relapse']==True].copy()
        non_relapse = ret_df[ret_df['relapse']==False].copy()

        #Raise errors if dataframes are too small to sample all wanted images
        if (num_relapse > len(relapse)) & (num_non_relapse > len(non_relapse)):
            raise Exception('Not enough images in relapse and non-relapse sets. Number of non-relapse images: ' 
                            + str(len(non_relapse))+ '. Number of relapse images: '+str(len(relapse)))
        
        if num_relapse > len(relapse):
            raise Exception('Not enough images in relapse set. Number of relapse images: '+str(len(relapse)))
        
        elif num_non_relapse > len(non_relapse):
            raise Exception('Not enough images in non_relapse set. Number of non_relapse images: '+str(len(non_relapse)))
        
        #sample wanted number of images
        if not num_relapse == len(relapse):
            relapse = relapse.sample(n=num_relapse, random_state=seed).reset_index(drop=True)
        if not num_non_relapse == len(non_relapse):
            non_relapse = non_relapse.sample(n=num_non_relapse, random_state=seed).reset_index(drop=True)
        ret_df = pd.concat([relapse,non_relapse], ignore_index=True)
    
    # Sample death if needed
    if (num_death + num_alive)> 0:
        death = ret_df[ret_df['Cause_of_Death'] == 'MUNUAISSYÖPÄ'].copy()
        alive = ret_df[ret_df['Cause_of_Death'] != 'MUNUAISSYÖPÄ'].copy()

        #Raise errors if dataframes are too small to sample all wanted images
        if (num_death > len(death)) & (num_alive > len(alive)):
            raise Exception('Not enough images in death and alive sets. Number of alive images: ' 
                            + str(len(alive))+ '. Number of death images: '+str(len(death)))
        
        if num_death > len(death):
            raise Exception('Not enough images in death set. Number of death images: '+str(len(death)))
        
        elif num_alive > len(alive):
            raise Exception('Not enough images in alive set. Number of alive images: '+str(len(alive)))
        
        #sample wanted number of images
        if not num_death == len(death):
            death = death.sample(n=num_death, random_state=seed).reset_index(drop=True)
        if not num_alive == len(alive):
            alive = alive.sample(n=num_alive, random_state=seed).reset_index(drop=True)
        ret_df = pd.concat([death,alive], ignore_index=True)
    
    return ret_df