import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.decomposition import PCA

NAN_GRID = {
    'drop_features' : ['line', 'name', 'mold_name', 'time', 'date',
                       'emergency_stop', 'molten_volume', 'registration_time'],
    'simple_fill_dict' : {'tryshot_signal' : 'No', 'heating_furnace' : 'C'},
    'mode_fill_features' : ['upper_mold_temp3', 'lower_mold_temp3', 'molten_temp'],
    'mode_criterion' : 'mold_code'
}

ENCODE_GRID = {
    'working' : ['정지', '가동'],
    'tryshot_signal' : ['No', 'D'],
    'heating_furnace' : ['A', 'B', 'C'],
    'mold_code' : [8412, 8413, 8573, 8576, 8600, 8722, 8917]
}

def load_data(path):
    data_configs = {}

    data = pd.read_csv(path, encoding='cp949', index_col=0, low_memory=False)
    
    data_configs['data'] = data
    data_configs['numeric_features'] = data.select_dtypes(['int64', 'float64']).columns
    data_configs['object_features'] = data.select_dtypes(['object']).columns

    return data_configs

def check_fail_rate(data):
    num_pass = len(data[data['passorfail'] == 0])
    num_fail = len(data[data['passorfail'] == 1])

    print(f"합격 데이터 수 : {num_pass}")
    print(f"불합격 데이터 수 : {num_fail}")
    print(f"불합격 데이터 비율 : {round(num_fail / (num_pass + num_fail) * 100, 2)} %")

def check_nan(data, percent=False):
    if percent:
        print(data.isna().sum() / len(data))
    else:
        print(data.isna().sum())

class NanProcessor:
    def __init__(self, nan_grid):
        self.drop_features = nan_grid['drop_features']
        self.simple_fill_dict = nan_grid['simple_fill_dict']
        self.mode_fill_features = nan_grid['mode_fill_features']
        self.mode_criterion = nan_grid['mode_criterion']
    
    def process(self, data):
        data = data.dropna(subset=['passorfail'])

        for feature, fill_val in self.simple_fill_dict.items():
            if feature == 'heating_furnace':
                condition = (data[feature].isna()) & (data['molten_volume'].isna())
                data.loc[condition, feature] = data.loc[condition, feature].fillna('D').astype('object')
            data.loc[:,feature] = data.loc[:, feature].fillna(fill_val)
        
        condition = ~(data['heating_furnace'] == 'D')
        data = data.loc[condition, :]
        
        data = data.drop(columns=self.drop_features)
        
        for feature in self.mode_fill_features:
            data[feature] = data.groupby(self.mode_criterion)[feature].transform(
                lambda x : x.fillna(x.mode()[0] if not x.mode().empty else x.mean())
            )
        
        data = data.reset_index(drop=True)

        return data

class CatFeatureEncoder:
    def __init__(self, encode_grid):
        self.encode_grid = encode_grid
    
    def process(self, data):
        for feature, ordinal in self.encode_grid.items():
            encoder = OrdinalEncoder(categories=[ordinal])
            data[[feature]] = encoder.fit_transform(data[[feature]])
        
        return data

def remove_outlier_by_lof(data, features):
    for feature in features:
        lof = LocalOutlierFactor(n_neighbors=10, n_jobs=-1)

        y_pred = lof.fit_predict(data[[feature]])
    
        data['outlier'] = y_pred
        
        data = data[data['outlier'] == 1].reset_index(drop=True)

        data = data.drop(columns=['outlier'])
    
    return data

class T_Testor:
    def __init__(self, p_threshold=0.05):
        self.p_threshold = p_threshold

    def test(self, data):
        self.t_test_configs = {}

        self.t_test = []
        self.useful_features = []

        t_test_features = data.columns
        t_test_features = [feature for feature in t_test_features if feature != 'passorfail']

        for feature in t_test_features:
            t=scipy.stats.ttest_ind(data[data['passorfail']==1][feature], 
                                    data[data['passorfail']==0][feature],
                                    equal_var=False)
            
            self.t_test.append([feature, t[0], t[1]])
            
        self.t_test = pd.DataFrame(self.t_test, columns=['col', 'tvalue', 'pvalue'])
        
        for idx in range(len(self.t_test)):
            if self.t_test['pvalue'][idx] < self.p_threshold:
                self.useful_features.append(self.t_test['col'][idx])

        self.t_test_configs['t_test'] = self.t_test
        self.t_test_configs['useful_features'] = self.useful_features

        return self.t_test_configs
    
    def get_useful_data(self, data):
        return data[self.useful_features + ['passorfail']]

def remove_outliers_by_isoforest(data, outlier_rate=0.015):

    iso_forest = IsolationForest(
        n_estimators=500, 
        contamination=outlier_rate, 
        random_state=42
    )

    iso_forest.fit(data)

    pred = iso_forest.predict(data)

    outliers = data[pred == -1] # outliers
    normal = data[pred == 1] # normal

    cleaned_data = data[pred != -1]

    print(f"[Outlier-Remover Log] With Outliers Shape : {data.shape}")
    print(f"[Outlier-Remover Log] Without Outliers Shape : {cleaned_data.shape}")

    return cleaned_data


class DataResampler:
    def __init__(self, downsampled_pass_rate, upsampled_fail_rate_about_pass, upsample_method='smote', with_pca=False):
        self.downsampled_pass_rate = downsampled_pass_rate
        self.upsampled_fail_rate_about_pass = upsampled_fail_rate_about_pass
        self.upsample_method = upsample_method
        self.with_pca = with_pca
    
    def process(self, train_data, train_label):
        # train_data = train_data.reset_index(drop=True)
        # test_data = test_data.reset_index(drop=True)
        # train_label = train_label.reset_index(drop=True)
        # test_label = test_label.reset_index(drop=True)

        fail_data = train_data[train_label == 1]
        pass_data = train_data[train_label == 0]

        if self.with_pca:
            downsampled_pass_data = resample(
                pass_data,
                replace = False,
                n_samples = round(len(pass_data) * self.downsampled_pass_rate),
                random_state = 42
            )
        else:
            downsampled_pass_data = resample(
                pass_data,
                replace = False,
                n_samples = round(len(pass_data) * self.downsampled_pass_rate),
                random_state = 42,
                stratify = pass_data['mold_code']
            )
        downsampled_pass_label = train_label[downsampled_pass_data.index]

        # not_used_pass_data = pass_data.drop(downsampled_pass_data.index)
        # not_used_pass_label = train_label[pass_data.index].drop(downsampled_pass_data.index)

        downsampled_pass_data = downsampled_pass_data.reset_index(drop=True)
        downsampled_pass_label = downsampled_pass_label.reset_index(drop=True)
        # not_used_pass_data = not_used_pass_data.reset_index(drop=True)
        # not_used_pass_label = not_used_pass_label.reset_index(drop=True)

        # test_data = pd.concat([test_data, not_used_pass_data], axis=0).reset_index(drop=True)
        # test_label = pd.concat([test_label, not_used_pass_label], axis=0).reset_index(drop=True)

        train_data = pd.concat([fail_data, downsampled_pass_data], axis=0).reset_index(drop=True)
        train_label = pd.concat([train_label[fail_data.index], downsampled_pass_label], axis=0).reset_index(drop=True)

        if self.upsample_method == 'smote':
            smote = SMOTE(sampling_strategy=self.upsampled_fail_rate_about_pass,
                          random_state=42)
            train_data, train_label = smote.fit_resample(train_data, train_label)

        elif self.upsample_method == 'adasyn':
            adasyn = ADASYN(sampling_strategy=self.upsampled_fail_rate_about_pass,
                            n_neighbors=10,
                            random_state=42)
            train_data, train_label = adasyn.fit_resample(train_data, train_label)


        return train_data, train_label



class FeatureEngineer:
    def __init__(self, do_count_trend=True, drop_count=True):
        self.do_count_trend=do_count_trend
        self.drop_count = drop_count
    
    def get_count_trend_feature(self, data):
        count_trend = []

        for count in data['count']:
            if (count >= 1) and (count <= 5):
                count_trend.append(2)
            elif (count >= 6) and (count <= 10):
                count_trend.append(1)
            else:
                count_trend.append(0)

        data['count_trend'] = count_trend

        if self.drop_count:
            data = data.drop(columns=['count'])

        return data
    
    def process(self, data):
        if self.do_count_trend:
            data = self.get_count_trend_feature(data)
        
        return data


class PCAProcessor:
    def __init__(self, variance_rate):
        self.variance_rate = variance_rate

        self.pca_computer = PCA()
    
    def process(self, data):
        self.pca_computer.fit(data)

        explained_variance_ratio = self.pca_computer.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        n_components = np.argmax(cumulative_variance >= self.variance_rate) + 1

        self.pca_computer = PCA(n_components=n_components)

        pca_result = self.pca_computer.fit_transform(data)
        pca_result = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])

        return pca_result


class KampDataLoader:
    def __init__(self, 
                 path,
                 
                 nan_grid=NAN_GRID, 

                 do_count_trend=True,
                 drop_count=True,
                 
                 encode_grid=ENCODE_GRID, 
                 
                 outlier_method='iso',
                 iso_outlier_rate=0.015,

                 p_threshold=0.05, 
                 get_useful_p_data=False, 

                 do_resample=True,
                 downsampled_pass_rate=0.5, 
                 upsampled_fail_rate_about_pass=1.0,
                 upsample_method='smote',
                 
                 do_pca=False,
                 variance_rate=0.95,
                 scale_include_cat=True):
        '''
        [Parameters]
        1. path 
            - Raw data 경로
        2. nan_grid (default=NAN_GRID) 
            - 결측치 처리를 위한 설정값 Dict
            - 재설정 필요 시 'NAN_GRID'를 Import 하여 변경 후 재설정
        3. encode_grid (default=NAN_GRID)
            - 범주형 feature encoding을 위한 설정값 Dict
            - 재설정 필요 시 'NAN_GRID'를 Import 하여 변경 후 재설정
        4. do_count_trend (defaut=True)
            - 'count' 변수를 구간화 하는 feature를 추가할 지 여부
        5. drop_count (default=True)
            - 'count_trend' 변수 생성 후 기존 'count' 변수를 drop할 지 여부
        6. outlier_method (defaul='iso')
            - 이상치 제거 방식 지정
            - 'iso' : 'isolation-forest' 방식 사용
            - 'lof' : 'LOF' 방식 사용
        7. iso_outlier_rate (default=0.015)
            - 'iso' 방식으로 이상치 제거 시 설정할 전체 데이터에서의 예상되는 이상치 비율
        8. p_threshold (default=0.05)
            - 의미 있는 feature를 구분할 p-value 기준 값
        9. get_useful_p_data (default=False)
            - p-value가 p_threshold의 설정값보다 높은 feature만 남길 지 여부
        10. do_resample
            - 클래스 불균형 해결을 위한 resampling을 진행할 지 여부
        11. downsampling_pass_rate (default=0.5)
            - 전체 데이터 중 pass 클래스인 데이터에 대해 남길 pass 데이터의 비율
            - 1.0으로 생성 시 모든 pass 데이터 생성
        12. upsample_fail_rate_about_pass (default='1.0')
            - downsampling 이후 upsampling 진행 시 pass 데이터에 대해 fail 데이터가 차지할 비율
            - 1.0으로 설정 시 pass 데이터와 동일한 양까지 생성
        13. upsample_method (default='smote')
            - upsampling을 진행할 방식 지정
        14. do_pca (defalut=False)
            - pca를 진행할 지 여부
        15. varance_rate (default=0.95)
            - pca 진행 시 선정된 주성분이 설명해야할 전체 분산에서의 비율
        16. scale_include_cat (default=True)
            - scaling 적용 시 '범주형 변수'도 포함하여 스케일링 할 지 여부
        '''
        
        self.path = path

        self.do_count_trend = do_count_trend
        self.drop_count = drop_count

        self.nan_grid = nan_grid
        
        self.encode_grid = encode_grid
        
        self.iso_outlier_rate = iso_outlier_rate
        self.outlier_method=outlier_method

        self.p_threshold = p_threshold
        self.get_useful_p_data = get_useful_p_data

        self.do_resample = do_resample
        self.downsampled_pass_rate = downsampled_pass_rate
        self.upsampled_fail_rate_about_pass = upsampled_fail_rate_about_pass
        self.upsample_method = upsample_method

        self.do_pca = do_pca
        self.variance_rate = variance_rate

        self.scale_include_cat = scale_include_cat
    
    def process(self):
        print('='*20, '[Data Process Start]', '='*20, '\n')

        # 로우 데이터 로드
        print("[Process Log] Loading Raw Data...")
        data_configs = load_data(self.path)
        print("[Process Log] Done\n")

        # 데이터 configs 설정
        data = data_configs['data']
        numeric_features = data_configs['numeric_features']
        object_features = data_configs['object_features']

        # 결측치 처리
        print("[Process Log] Processing Nan Value...")
        data = NanProcessor(nan_grid=self.nan_grid).process(data)
        print("[Process Log] Done\n")

        if self.do_count_trend:
            print("[Process Log] Feature Engineering...")
            data = FeatureEngineer(do_count_trend=self.do_count_trend, 
                                   drop_count=self.drop_count).process(data)
            print("[Process Log] Done\n")

        # 범주형 변수 인코딩
        print("[Process Log] Encoding Categorical Features...")
        data = CatFeatureEncoder(encode_grid=self.encode_grid).process(data)
        print("[Process Log] Done\n")

        # 이상치 처리
        # IsolationForest 방식
        if self.outlier_method == 'iso':
            print("[Process Log] Removing Outliers (IsoForest)...")
            data = remove_outliers_by_isoforest(data=data, outlier_rate=self.iso_outlier_rate)
            print("[Process Log] Done\n")
        # LOF 방식
        elif self.outlier_method == 'lof':
            print("[Process Log] Removing Outliers (LOF)...")
            numeric_features = [feature for feature in numeric_features 
                        if feature not in ['count', 'molten_volume', 'mold_code']]
            data = remove_outlier_by_lof(data, numeric_features)
            print("[Process Log] Done\n")

        # T-Test 기반 feauture 선정
        if self.get_useful_p_data:
            print("[Process Log] T-Testing...")
            t_test = T_Testor(p_threshold=self.p_threshold)
            t_test_configs = t_test.test(data)
            data = t_test.get_useful_data(data)
            print("[Process Log] Done\n")
        
        # 데이터 스케일링 (MinMaxScaler)
        if not self.scale_include_cat:
            if self.do_count_trend:
                print("[Process Log] Data Scaling (MinMaxScaler)...")
                cat_data = data[['working', 'EMS_operation_time', 'mold_code', 'heating_furnace', 'count_trend']]
                data_input = data.drop(columns=['passorfail', 'working', 'EMS_operation_time', 'mold_code', 'heating_furnace', 'count_trend'])
            else:
                print("[Process Log] Data Scaling (MinMaxScaler)...")
                cat_data = data[['working', 'EMS_operation_time', 'mold_code', 'heating_furnace']]
                data_input = data.drop(columns=['passorfail', 'working', 'EMS_operation_time', 'mold_code', 'heating_furnace'])     
            cat_data = cat_data.reset_index(drop=True)
            data_input = data_input.reset_index(drop=True)
            input_feature_names = data_input.columns
            data_label = data['passorfail']
            scaler = MinMaxScaler()
            data_input = scaler.fit_transform(data_input)
            data_input = pd.DataFrame(data_input, columns=input_feature_names)
            data_input = pd.concat([data_input, cat_data], axis=1)
            print("[Process Log] Done\n")
        else:
            print("[Process Log] Data Scaling (MinMaxScaler)...")
            data_input = data.reset_index(drop=True)
            input_feature_names = data_input.columns
            data_label = data['passorfail']
            scaler = MinMaxScaler()
            data_input = scaler.fit_transform(data_input)
            data_input = pd.DataFrame(data_input, columns=input_feature_names)
            print("[Process Log] Done\n")

        # PCA 진행
        if self.do_pca:
            print("[Process Log] PCA..")
            data_input = PCAProcessor(variance_rate=self.variance_rate).process(data=data)
            print("[Process Log] Done\n")
        

        # 학습-평가 데이터 분할
        print("[Process Log] Train Test Spliting...")
        train_data, test_data, train_label, test_label = train_test_split(
            data_input, data_label,
            test_size=0.2,
            stratify=data_label,
            random_state=42
        )
        print("[Process Log] Done\n")

        train_data = train_data.reset_index(drop=True)
        train_label = train_label.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
        test_label = test_label.reset_index(drop=True)

        # 데이터 리샘플링
        if self.do_resample:
            print(f"[Process Log] Data Resampling ({self.upsample_method})...")
            train_data, train_label = DataResampler(downsampled_pass_rate=self.downsampled_pass_rate,
                                                             upsampled_fail_rate_about_pass=self.upsampled_fail_rate_about_pass,
                                                             upsample_method=self.upsample_method,
                                                             with_pca=self.do_pca).process(train_data, train_label)
            print("[Process Log] Done\n")
        
        # if self.do_count_trend:
        #     remapping_features = ['working', 'EMS_operation_time', 'mold_code', 'heating_furnace', 'count_trend']
        #     for feature in remapping_features:
        #         train_data[feature] = train_data[feature].apply(lambda x : round(x))
        #         test_data[feature] = test_data[feature].apply(lambda x : round(x))
        # else:
        #     remapping_features = ['working', 'EMS_operation_time', 'mold_code', 'heating_furnace']
        #     for feature in remapping_features:
        #         train_data[feature] = train_data[feature].apply(lambda x : round(x))
        #         test_data[feature] = test_data[feature].apply(lambda x : round(x))

        self.data = {
            'train_data' : train_data,
            'train_label' : train_label,
            'test_data' : test_data,
            'test_label' : test_label
        }

        print('='*23, '[Done]', '='*23)

    def load(self):
        return self.data