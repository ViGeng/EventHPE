import pickle
with open ('/root/EventHPE/data_event/data_event_out/pose_events/subject01_group1_time1/pose_info.pkl', 'rb') as fp:
    test_joints = pickle.load(fp)
    print(type(test_joints))