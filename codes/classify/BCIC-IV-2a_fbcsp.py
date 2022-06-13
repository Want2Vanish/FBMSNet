import numpy as np
import scipy.signal as signal
from scipy.signal import cheb2ord

import numpy as np

class Classifier:
    def __init__(self,model):
        self.model = model
        self.feature_selection = False

    def predict(self,x_features):
        if self.feature_selection:
            x_features_selected = self.feature_selection.transform(x_features)
        else:
            x_features_selected = x_features
        y_predicted = self.model.predict(x_features_selected)
        return y_predicted

    def fit(self,x_features,y_train):
        feature_selection = True
        if feature_selection:
            feature_selection = FeatureSelect()
            self.feature_selection = feature_selection
            x_train_features_selected = self.feature_selection.fit(x_features,y_train)
        else:
            x_train_features_selected = x_features
        self.model.fit(x_train_features_selected,y_train)
        y_predicted = self.model.predict(x_train_features_selected)
        return y_predicted


class FeatureSelect:
    def __init__(self, n_features_select=4, n_csp_pairs=2):
        self.n_features_select = n_features_select
        self.n_csp_pairs = n_csp_pairs
        self.features_selected_indices=[]

    def fit(self,x_train_features,y_train):
        MI_features = self.MIBIF(x_train_features, y_train)
        MI_sorted_idx = np.argsort(MI_features)[::-1]
        features_selected = MI_sorted_idx[:self.n_features_select]

        paired_features_idx = self.select_CSP_pairs(features_selected, self.n_csp_pairs)
        x_train_features_selected = x_train_features[:, paired_features_idx]
        self.features_selected_indices = paired_features_idx

        return x_train_features_selected

    def transform(self,x_test_features):
        return x_test_features[:,self.features_selected_indices]

    def MIBIF(self, x_features, y_labels):
        def get_prob_pw(x,d,i,h):
            n_data = d.shape[0]
            t=d[:,i]
            kernel = lambda u: np.exp(-0.5*(u**2))/np.sqrt(2*np.pi)
            prob_x = 1 / (n_data * h) * sum(kernel((np.ones((len(t)))*x- t)/h))
            return prob_x

        def get_pd_pw(d, i, x_trials):
            n_data, n_dimensions = d.shape
            if n_dimensions==1:
                i=1
            t = d[:,i]
            min_x = np.min(t)
            max_x = np.max(t)
            n_trials = x_trials.shape[0]
            std_t = np.std(t)
            if std_t==0:
                h=0.005
            else:
                h=(4./(3*n_data))**(0.2)*std_t
            prob_x = np.zeros((n_trials))
            for j in range(n_trials):
                prob_x[j] = get_prob_pw(x_trials[j],d,i,h)
            return prob_x, x_trials, h

        y_classes = np.unique(y_labels)
        n_classes = len(y_classes)
        n_trials = len(y_labels)
        prob_w = []
        x_cls = {}
        for i in range(n_classes):
            cls = y_classes[i]
            cls_indx = np.where(y_labels == cls)[0]
            prob_w.append(len(cls_indx) / n_trials)
            x_cls.update({i: x_features[cls_indx, :]})

        prob_x_w = np.zeros((n_classes, n_trials, x_features.shape[1]))
        prob_w_x = np.zeros((n_classes, n_trials, x_features.shape[1]))
        h_w_x = np.zeros((x_features.shape[1]))
        mutual_info = np.zeros((x_features.shape[1]))
        parz_win_width = 1.0 / np.log2(n_trials)
        h_w = -np.sum(prob_w * np.log2(prob_w))

        for i in range(x_features.shape[1]):
            h_w_x[i] = 0
            for j in range(n_classes):
                prob_x_w[j, :, i] = get_pd_pw(x_cls.get(j), i, x_features[:, i])[0]

        t_s = prob_x_w.shape
        n_prob_w_x = np.zeros((n_classes, t_s[1], t_s[2]))
        for i in range(n_classes):
            n_prob_w_x[i, :, :] = prob_x_w[i] * prob_w[i]
        prob_x = np.sum(n_prob_w_x, axis=0)
        # prob_w_x = np.zeros((n_classes, prob_x.shape[0], prob_w.shape[1]))
        for i in range(n_classes):
            prob_w_x[i, :, :] = n_prob_w_x[i, :, :]/prob_x

        for i in range(x_features.shape[1]):
            for j in range(n_trials):
                t_sum = 0.0
                for k in range(n_classes):
                    if prob_w_x[k, j, i] > 0:
                        t_sum += (prob_w_x[k, j, i] * np.log2(prob_w_x[k, j, i]))

                h_w_x[i] -= (t_sum / n_trials)

            mutual_info[i] = h_w - h_w_x[i]

        mifsg = np.asarray(mutual_info)
        return mifsg


    def select_CSP_pairs(self,features_selected,n_pairs):
        features_selected+=1
        sel_groups = np.unique(np.ceil(features_selected/n_pairs))
        paired_features = []
        for i in range(len(sel_groups)):
            for j in range(n_pairs-1,-1,-1):
                paired_features.append(sel_groups[i]*n_pairs-j)

        paired_features = np.asarray(paired_features,dtype=np.int)-1

        return paired_features

from sklearn.svm import SVR
import mne
import scipy.linalg
import numpy as np

class CSP:
    def __init__(self,m_filters):
        self.m_filters = m_filters

    def fit(self,x_train,y_train):
        x_data = np.copy(x_train)
        y_labels = np.copy(y_train)
        n_trials, n_channels, n_samples = x_data.shape
        cov_x = np.zeros((2, n_channels, n_channels), dtype=np.float)
        for i in range(n_trials):
            x_trial = x_data[i, :, :]
            y_trial = y_labels[i]
            cov_x_trial = np.matmul(x_trial, np.transpose(x_trial))
            cov_x_trial /= np.trace(cov_x_trial)
            cov_x[y_trial, :, :] += cov_x_trial

        cov_x = np.asarray([cov_x[cls]/np.sum(y_labels==cls) for cls in range(2)])
        cov_combined = cov_x[0]+cov_x[1]
        eig_values, u_mat = scipy.linalg.eig(cov_combined,cov_x[0])
        sort_indices = np.argsort(abs(eig_values))[::-1]
        eig_values = eig_values[sort_indices]
        u_mat = u_mat[:,sort_indices]
        u_mat = np.transpose(u_mat)

        return eig_values, u_mat

    def transform(self,x_trial,eig_vectors):
        z_trial = np.matmul(eig_vectors, x_trial)
        z_trial_selected = z_trial[:self.m_filters,:]
        z_trial_selected = np.append(z_trial_selected,z_trial[-self.m_filters:,:],axis=0)
        sum_z2 = np.sum(z_trial_selected**2, axis=1)
        sum_z = np.sum(z_trial_selected, axis=1)
        var_z = (sum_z2 - (sum_z ** 2)/z_trial_selected.shape[1]) / (z_trial_selected.shape[1] - 1)
        sum_var_z = sum(var_z)
        return np.log(var_z/sum_var_z)
import mne
import os
import glob
import numpy as np
from scipy.io import loadmat, savemat

class LoadData:
    def __init__(self,eeg_file_path: str):
        self.eeg_file_path = eeg_file_path

    def load_raw_data_gdf(self,file_to_load):
        self.raw_eeg_subject = mne.io.read_raw_gdf(self.eeg_file_path + '/' + file_to_load)
        return self

    def load_raw_data_mat(self,file_to_load):
        import scipy.io as sio
        self.raw_eeg_subject = sio.loadmat(self.eeg_file_path + '/' + file_to_load)

    def get_all_files(self,file_path_extension: str =''):
        if file_path_extension:
            return glob.glob(self.eeg_file_path+'/'+file_path_extension)
        return os.listdir(self.eeg_file_path)

class LoadBCIC(LoadData):
    '''Subclass of LoadData for loading BCI Competition IV Dataset 2a'''
    def __init__(self,*args):
        super(LoadBCIC,self).__init__(*args)

    def get_epochs(self, dataPath, labelPath, epochWindow = [0,4], chans = list(range(22))):
        eventCode = [2]  # start of the trial at t=0
        fs = 250
        offset = 2

        # load the gdf file using MNE
        raw_gdf = mne.io.read_raw_gdf(dataPath, stim_channel="auto")
        raw_gdf.load_data()
        gdf_events = mne.events_from_annotations(raw_gdf)[0][:, [0, 2]].tolist()
        eeg = raw_gdf.get_data()

        # drop channels
        if chans is not None:
            eeg = eeg[chans, :]

        # Epoch the data
        events = [event for event in gdf_events if event[1] in eventCode]
        y = np.array([i[1] for i in events])
        epochInterval = np.array(range(epochWindow[0] * fs, epochWindow[1] * fs)) + offset * fs
        x = np.stack([eeg[:, epochInterval + event[0]] for event in events], axis=2)

        # Multiply the data with 1e6
        x = x * 1e6

        # Load the labels

        y = loadmat(labelPath)["classlabel"].squeeze()
        # change the labels from [1-4] to [0-3]
        y = y - 1

        data = {'x': x, 'y': y, 'c': np.array(raw_gdf.info['ch_names'])[chans].tolist(), 's': raw_gdf.info.get('sfreq')}
        return data

class LoadKU(LoadData):
    '''Subclass of LoadData for loading KU Dataset'''
    def __init__(self,subject_id,*args):
        self.subject_id=subject_id
        self.fs=1000
        super(LoadKU,self).__init__(*args)

    def get_epochs(self,sessions=[1, 2]):
        for i in sessions:
            file_to_load=f'session{str(i)}/s{str(self.subject_id)}/EEG_MI.mat'
            self.load_raw_data_mat(file_to_load)
            x_data = self.raw_eeg_subject['EEG_MI_train']['smt'][0, 0]
            x_data = np.transpose(x_data,axes=[1, 2, 0])
            labels = self.raw_eeg_subject['EEG_MI_train']['y_dec'][0, 0][0]
            y_labels = labels - np.min(labels)
            if hasattr(self,'x_data'):
                self.x_data=np.append(self.x_data,x_data,axis=0)
                self.y_labels=np.append(self.y_labels,y_labels)
            else:
                self.x_data = x_data
                self.y_labels = y_labels
        ch_names = self.raw_eeg_subject['EEG_MI_train']['chan'][0, 0][0]
        ch_names_list = [str(x[0]) for x in ch_names]
        eeg_data = {'x_data': self.x_data,
                    'y_labels': self.y_labels,
                    'fs': self.fs,
                    'ch_names':ch_names_list}

        return eeg_data

class PreprocessKU:
    def __init__(self):
        self.selected_channels=['FC5','FC3','FC1','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6']

    def select_channels(self,x_data,ch_names,selected_channels=[]):
        if not selected_channels:
            selected_channels = self.selected_channels
        selected_channels_idx = mne.pick_channels(ch_names, selected_channels,[])
        x_data_selected = x_data[:,selected_channels_idx,:].copy()
        return x_data_selected

class FBCSP:
    def __init__(self,m_filters):
        self.m_filters = m_filters
        self.fbcsp_filters_multi=[]

    def fit(self,x_train_fb,y_train):
        y_classes_unique = np.unique(y_train)
        n_classes = len(y_classes_unique)
        self.csp = CSP(self.m_filters)

        def get_csp(x_train_fb, y_train_cls):
            fbcsp_filters = {}
            for j in range(x_train_fb.shape[0]):
                x_train = x_train_fb[j, :, :, :]
                eig_values, u_mat = self.csp.fit(x_train, y_train_cls)
                fbcsp_filters.update({j: {'eig_val': eig_values, 'u_mat': u_mat}})
            return fbcsp_filters

        for i in range(n_classes):
            cls_of_interest = y_classes_unique[i]
            select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]
            y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))
            fbcsp_filters=get_csp(x_train_fb,y_train_cls)
            self.fbcsp_filters_multi.append(fbcsp_filters)

    def transform(self,x_data,class_idx=0):
        n_fbanks, n_trials, n_channels, n_samples = x_data.shape
        x_features = np.zeros((n_trials,self.m_filters*2*len(x_data)),dtype=np.float)
        for i in range(n_fbanks):
            eig_vectors = self.fbcsp_filters_multi[class_idx].get(i).get('u_mat')
            eig_values = self.fbcsp_filters_multi[class_idx].get(i).get('eig_val')
            for k in range(n_trials):
                x_trial = np.copy(x_data[i,k,:,:])
                csp_feat = self.csp.transform(x_trial,eig_vectors)
                for j in range(self.m_filters):
                    x_features[k, i * self.m_filters * 2 + (j+1) * 2 - 2]  = csp_feat[j]
                    x_features[k, i * self.m_filters * 2 + (j+1) * 2 - 1]= csp_feat[-j-1]

        return x_features


class MLEngine:
    def __init__(self,datasetPath='',subject_id='',sessions=[1, 2],ntimes=1,kfold=2,m_filters=2,window_details={}):
        self.datasetPath = datasetPath
        self.subject_id=subject_id
        self.sessions = sessions
        self.kfold = kfold
        self.ntimes=ntimes
        self.window_details = window_details
        self.m_filters = m_filters

    def experiment(self):

        '''for BCIC Dataset'''
        datasetPath = 'D:/Projects/Pycharm/helloPycharm/FBCNet-master/data/bci42a/originalData'
        subjects = ['A01T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T', 'A07T', 'A08T', 'A09T']
        test_subjects = ['A01E', 'A02E', 'A03E', 'A04E', 'A05E', 'A06E', 'A07E', 'A08E', 'A09E']

        bcic_data = LoadBCIC('1')
        training_accuracy = []
        testing_accuracy = []
        for i in range(len(subjects)):
            train_set = bcic_data.get_epochs(os.path.join(datasetPath, subjects[i] + '.gdf'),
                                            os.path.join(datasetPath, subjects[i] +'.mat'))
            test_set = bcic_data.get_epochs(os.path.join(datasetPath, test_subjects[i] + '.gdf'),
                                            os.path.join(datasetPath, test_subjects[i] + '.mat'))
            train_data = train_set['x'].transpose(2,0,1)
            train_label = train_set['y']
            test_data = test_set['x'].transpose(2,0,1)
            test_label = test_set['y']

            trainrate = 1.0
            train_data = train_data[:int(len(train_label)*trainrate)]
            train_label = train_label[:int(len(train_label)*trainrate)]
            print("hahahahah: ***********",int(len(train_label)))

            fbank = FilterBank(250)
            fbank_coeff = fbank.get_filter_coeff()
            filtered_train_data = fbank.filter_data(train_data,self.window_details)
            filtered_test_data = fbank.filter_data(test_data, self.window_details)

            y_classes_unique = np.unique(train_label)
            n_classes = len(np.unique(train_label))

            fbcsp = FBCSP(self.m_filters)
            fbcsp.fit(filtered_train_data, train_label)
            y_train_predicted = np.zeros((train_label.shape[0], n_classes), dtype=np.float)
            y_test_predicted = np.zeros((test_label.shape[0], n_classes), dtype=np.float)

            for j in range(n_classes):
                cls_of_interest = y_classes_unique[j]
                select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]

                y_train_cls = np.asarray(select_class_labels(cls_of_interest, train_label))
                y_test_cls = np.asarray(select_class_labels(cls_of_interest, test_label))

                x_features_train = fbcsp.transform(filtered_train_data, class_idx=cls_of_interest)
                x_features_test = fbcsp.transform(filtered_test_data, class_idx=cls_of_interest)

                classifier_type = SVR(gamma='auto')
                classifier = Classifier(classifier_type)
                y_train_predicted[:, j] = classifier.fit(x_features_train, np.asarray(y_train_cls, dtype=np.float))
                y_test_predicted[:, j] = classifier.predict(x_features_test)

            y_train_predicted_multi = self.get_multi_class_regressed(y_train_predicted)
            y_test_predicted_multi = self.get_multi_class_regressed(y_test_predicted)
            from sklearn.metrics import confusion_matrix
            print(confusion_matrix(test_label, y_test_predicted_multi))
            tr_acc = np.sum(y_train_predicted_multi == train_label, dtype=np.float) / len(train_label)
            te_acc = np.sum(y_test_predicted_multi == test_label, dtype=np.float) / len(test_label)

            print(f'Training Accuracy = {str(tr_acc)}\n')
            print(f'Testing Accuracy = {str(te_acc)}\n \n')

            training_accuracy.append(tr_acc)
            testing_accuracy.append(te_acc)

        mean_training_accuracy = np.mean(np.asarray(training_accuracy))
        mean_testing_accuracy = np.mean(np.asarray(testing_accuracy))
        std_testing_accuracy = np.std(np.asarray(testing_accuracy))
        print('*' * 10)
        print(f'Mean Training Accuracy = {str(mean_training_accuracy)}')
        print(f'Mean Testing Accuracy = {str(mean_testing_accuracy)}')
        print(f'Std Testing Accuracy = {str(std_testing_accuracy)}')
        print(testing_accuracy)
        print('*' * 10)

    def cross_validate_Ntimes_Kfold(self, y_labels, ifold=0):
        from sklearn.model_selection import StratifiedKFold
        train_indices = {}
        test_indices = {}
        random_seed = ifold
        skf_model = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=random_seed)
        i = 0
        for train_idx, test_idx in skf_model.split(np.zeros(len(y_labels)), y_labels):
            train_indices.update({i: train_idx})
            test_indices.update({i: test_idx})
            i += 1
        return train_indices, test_indices

    def cross_validate_sequential_split(self, y_labels):
        from sklearn.model_selection import StratifiedKFold
        train_indices = {}
        test_indices = {}
        skf_model = StratifiedKFold(n_splits=self.kfold, shuffle=False)
        i = 0
        for train_idx, test_idx in skf_model.split(np.zeros(len(y_labels)), y_labels):
            train_indices.update({i: train_idx})
            test_indices.update({i: test_idx})
            i += 1
        return train_indices, test_indices

    def cross_validate_half_split(self, y_labels):
        import math
        unique_classes = np.unique(y_labels)
        all_labels = np.arange(len(y_labels))
        train_idx =np.array([])
        test_idx = np.array([])
        for cls in unique_classes:
            cls_indx = all_labels[np.where(y_labels==cls)]
            if len(train_idx)==0:
                train_idx = cls_indx[:math.ceil(len(cls_indx)/2)]
                test_idx = cls_indx[math.ceil(len(cls_indx)/2):]
            else:
                train_idx=np.append(train_idx,cls_indx[:math.ceil(len(cls_indx)/2)])
                test_idx=np.append(test_idx,cls_indx[math.ceil(len(cls_indx)/2):])

        train_indices = {0:train_idx}
        test_indices = {0:test_idx}

        return train_indices, test_indices

    def split_xdata(self,eeg_data, train_idx, test_idx):
        x_train_fb=np.copy(eeg_data[:,train_idx,:,:])
        x_test_fb=np.copy(eeg_data[:,test_idx,:,:])
        return x_train_fb, x_test_fb

    def split_ydata(self,y_true, train_idx, test_idx):
        y_train = np.copy(y_true[train_idx])
        y_test = np.copy(y_true[test_idx])

        return y_train, y_test

    def get_multi_class_label(self,y_predicted, cls_interest=0):
        y_predict_multi = np.zeros((y_predicted.shape[0]))
        for i in range(y_predicted.shape[0]):
            y_lab = y_predicted[i, :]
            lab_pos = np.where(y_lab == cls_interest)[0]
            if len(lab_pos) == 1:
                y_predict_multi[i] = lab_pos
            elif len(lab_pos > 1):
                y_predict_multi[i] = lab_pos[0]
        return y_predict_multi

    def get_multi_class_regressed(self, y_predicted):
        y_predict_multi = np.asarray([np.argmin(y_predicted[i,:]) for i in range(y_predicted.shape[0])])
        return y_predict_multi


class FilterBank:
    def __init__(self,fs):
        self.fs = fs
        self.f_trans = 2
        self.f_pass = np.arange(4,40,4)
        self.f_width = 4
        self.gpass = 3
        self.gstop = 30
        self.filter_coeff={}

    def get_filter_coeff(self):
        Nyquist_freq = self.fs/2

        for i, f_low_pass in enumerate(self.f_pass):
            f_pass = np.asarray([f_low_pass, f_low_pass+self.f_width])
            f_stop = np.asarray([f_pass[0]-self.f_trans, f_pass[1]+self.f_trans])
            wp = f_pass/Nyquist_freq
            ws = f_stop/Nyquist_freq
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a = signal.cheby2(order, self.gstop, ws, btype='bandpass')
            self.filter_coeff.update({i:{'b':b,'a':a}})

        return self.filter_coeff

    def filter_data(self,eeg_data,window_details={}):
        n_trials, n_channels, n_samples = eeg_data.shape
        # if window_details:
        #     n_samples = int(self.fs*(window_details.get('tmax')-window_details.get('tmin')))+1
        filtered_data=np.zeros((len(self.filter_coeff),n_trials,n_channels,n_samples))
        for i, fb in self.filter_coeff.items():
            b = fb.get('b')
            a = fb.get('a')
            eeg_data_filtered = np.asarray([signal.lfilter(b,a,eeg_data[j,:,:]) for j in range(n_trials)])
            filtered_data[i,:,:,:]=eeg_data_filtered
        return filtered_data

if __name__ == "__main__":
    '''Example for loading Korea University Dataset'''
    # dataset_details = {
    #     'data_path': "/Volumes/Transcend/BCI/KU_Dataset/BCI dataset/DB_mat",
    #     'subject_id': 1,
    #     'sessions': [1],
    #     'ntimes': 1,
    #     'kfold': 10,
    #     'm_filters': 2,
    # }

    '''Example for loading BCI Competition IV Dataset 2a'''
    dataset_details={
        'datasetPath' : '',
        'subject_id' : '',
        'ntimes': 1,
        'kfold':10,
        'm_filters':4,
        'window_details':{'tmin':0.0,'tmax':4}
    }

    ML_experiment = MLEngine(**dataset_details)
    ML_experiment.experiment()