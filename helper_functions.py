def makedir(path):
    """
    creates a directory if does not exist
    """
    if os.path.exists(path):
        print(path, "already exists")
    else:
        os.mkdir(path)
        print(path, "dir created")
        
        
def progress_update(start_time, completed_iters, total_iters):
    end_time = time.time()
    minute = round((end_time - start_time) / 60, 2)
    eta = (100 * minute) / (completed_iters/total_iters*100) # estimated time to complete
    eta = round(eta - minute, 2)
    
    print(f"{round(completed_iters/total_iters*100, 2)}% complete. {minute} minutes elapsed. Time left: {eta} minutes", end='\r')


def frame_to_int(frame_name):
    """
    convert frame names to integers. For example, frame_150.jpg becomes 150
    This is useful for sorting the frames
    """
    return int( frame_name[6:-4] )
    


def emotions_dictionary_to_df(emotions_dict):
    """
    convert the emotions_dict to a pandas df
    this is done because output format of deepface is difficult to use
    """
    
    frame_name = []
    angry = []
    disgust = []
    fear = []
    happy = []
    sad = []
    surprise = []
    neutral = []
    dominant_emotion = []
    x = []
    y = []
    w = []
    h = []


    for frame in emotions_dict:
        frame_name.append(frame)
        angry.append(emotions_dict[frame]['emotion']['angry'])
        disgust.append(emotions_dict[frame]['emotion']['disgust'])
        fear.append(emotions_dict[frame]['emotion']['fear'])
        happy.append(emotions_dict[frame]['emotion']['happy'])
        sad.append(emotions_dict[frame]['emotion']['sad'])
        surprise.append(emotions_dict[frame]['emotion']['surprise'])
        neutral.append(emotions_dict[frame]['emotion']['neutral'])

        dominant_emotion.append(emotions_dict[frame]['dominant_emotion'])

        x.append(emotions_dict[frame]['region']['x'])
        y.append(emotions_dict[frame]['region']['y'])
        w.append(emotions_dict[frame]['region']['w'])
        h.append(emotions_dict[frame]['region']['h'])
        
    df = pd.DataFrame(list(zip(frame_name, angry, disgust, fear, happy, sad, surprise, neutral, dominant_emotion,
        x, y, w, h)),
        columns =['frame_name','angry','disgust','fear','happy','sad','surprise','neutral','dominant_emotion',
        'x','y','w','h'])
    
    return df