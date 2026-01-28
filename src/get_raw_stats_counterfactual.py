from collections import defaultdict, Counter
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn.metrics import r2_score, log_loss
from sklearn.isotonic import IsotonicRegression
import os
import numpy as np
from tqdm import tqdm
import traceback

metric = (log_loss if os.environ.get('CELOSS', '0') == '1' else r2_score)

# metric = (
#     lambda y_true, y_pred: (-thres-log_loss(y_true, y_pred))/(-thres)) if os.environ.get('CELOSS', '0') == '1' else 
#     r2_score
# )

import json
import numpy as np
import random
np.random.seed(1900)
random.seed(2000)

def flip_score(score):
    thres = np.log(2)
    if os.environ.get('CELOSS', '0') == '1':
        return thres*2 - score
    return -score

def is_better(score1, score2):
    if os.environ.get('CELOSS', '0') == '1':
        return score1 < score2
    return score1 > score2

def get_score(X: np.ndarray, y: np.ndarray, silent=True):
    # Randomly permute the data and partition it into training and test sets
    assert not np.any(np.isnan(X))
    # print(len(X))
    indices = np.random.permutation(len(X)).tolist()
    train_size = int(0.6 * len(X))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    x_argmin = np.argmin(X)
    x_argmax = np.argmax(X)
    train_indices += [x_argmin, x_argmax]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    if not silent or os.environ.get('LINEAR', '0') == '1':
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrc = metric(y_test, y_pred)
        if model.coef_[0] < 0: metrc = flip_score(metrc)
        if not silent:
            print('metrc Results:')
            print(f'All types (Linear): {metrc:.5f}')
            print(f'All types (Linear, train): {metric(y_train, model.predict(X_train)):.5f}')

    if not silent or os.environ.get('POLY', '0') == '1':
        poly = PolynomialFeatures(degree=3)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.fit_transform(X_test)
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        y_pred_poly = model.predict(X_test_poly)
        metrc_poly = metric(y_test, y_pred_poly)
        if not silent:
            print(f'All types (Polynomial): {metrc_poly:.5f}')
            print(f'All types (Polynomial, train): {metric(y_train, model.predict(X_train_poly)):.5f}')

    if not silent or os.environ.get('LOGISTIC', '0') == '1':
        model_log = LogisticRegression(max_iter=1000)
        model_log.fit(X_train, y_train)
        y_pred_log = model_log.predict_proba(X_test)[:, 1]
        metrc_log = metric(y_test, y_pred_log)
        if model_log.coef_[0][0] < 0: metrc_log = flip_score(metrc_log)
        if not silent:
            print(f'Coefficients: {model_log.coef_}')
            print(f'All types (Logistic): {metrc_log:.5f}')
            print(f'All types (Logistic, train): {metric(y_train, model_log.predict_proba(X_train)[:, 1]):.5f}')
            print(f'Shapes: {X_train.shape, y_train.shape, X_test.shape, y_test.shape}')
            print(f'Train means: {np.mean(X_train.flatten()[y_train == 0]), np.mean(X_train.flatten()[y_train == 1])}')
            print(f'Test means: {np.mean(X_test.flatten()[y_test == 0]), np.mean(X_test.flatten()[y_test == 1])}')

    if not silent or os.environ.get('ISOTONIC', '0') == '1':
        model = IsotonicRegression(y_min=0, y_max=1)
        model.fit(X_train.flatten(), y_train)
        y_pred_iso = model.predict(X_test.flatten())
        metrc_iso = metric(y_test, y_pred_iso)
        if not silent:
            print(f'All types (Isotonic): {metrc_iso:.5f}')
            print(f'All types (Isotonic, train): {metric(y_train, model.predict(X_train.flatten())):.5f}')
        
        # Get reversed isotonic regression
        model = IsotonicRegression(y_min=0, y_max=1, increasing=False)
        model.fit(X_train.flatten(), y_train)
        y_pred_iso_rev = model.predict(X_test.flatten())
        metrc_iso_rev = metric(y_test, y_pred_iso_rev)
        if not silent:
            print(f'All types (Isotonic, rev): {metrc_iso_rev:.5f}')
    
    if not silent:
        from matplotlib import pyplot as plt
        plt.clf()
        ind = np.argsort(X_test.flatten())
        plt.scatter(X_test[ind], y_test[ind], color='black', label='True', s=5)
        plt.plot(X_test[ind], y_pred_iso[ind], color='red', label='Isotonic Fit')
        plt.plot(X_test[ind], y_pred[ind], color='blue', label='Linear Fit')
        plt.plot(X_test[ind], y_pred_poly[ind], color='green', label='Polynomial Fit')
        plt.plot(X_test[ind], y_pred_log[ind], color='orange', label='Logistic Fit')
        plt.legend()
        plt.show()
        plt.savefig('fit.pdf')
    
    if os.environ.get('LOGISTIC', '0') == '1':
        return metrc_log
    
    if os.environ.get('LINEAR', '0') == '1':
        return metrc
    
    if os.environ.get('ISOTONIC', '0') == '1':
        return metrc_iso if is_better(metrc_iso, metrc_iso_rev) else flip_score(metrc_iso_rev)
    
    if os.environ.get('POLY', '0') == '1':
        return metrc_poly
    
    assert False


def get_full(metrc: float, n: int, xs: list = None, ys: list = None):
    # Use bootstrapping to get the standard error of the metric
    assert xs is not None and ys is not None
    
    num_bootstraps = 1000
    bootstrapped_metrics = []
    print(len(xs), len(ys))
    assert len(xs) == len(ys) == n
    
    for _ in tqdm(range(num_bootstraps)):
        indices = np.random.permutation(len(xs)).tolist()
        bootstrapped_metrics.append(get_score(np.array(xs)[indices], np.array(ys)[indices]))
    
    bootstrapped_metrics = np.array(bootstrapped_metrics)
    
    return {
        'metric': np.mean(bootstrapped_metrics),
        'n': n,
        '90ci': np.percentile(bootstrapped_metrics, [10, 90]).tolist(),
    }
    

def get_id(name):
    if isinstance(name, int):
        return name

    for i in range(100, 0, -1):
        if f'b-{i}' in name.lower():
            return i - 1
    
    if '8b' in name.lower():
        return 0
    
    if '70b' in name.lower():
        return 1
    
    if '405b' in name.lower():
        return 2
    
    assert False

def get_metrc_for_llmasajudge(files, merge=False):
    expectation = defaultdict(lambda: 0)
    summation = defaultdict(lambda: 0)
    entries = defaultdict(lambda: [])
    entries_null = defaultdict(lambda: [])
    values_for_each = defaultdict(lambda: [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []])
    positions_for_each = defaultdict(lambda: [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []])
    
    vals0 = []
    valsn0 = []

    if not merge:
        for file, _ in files:
            with open(file, 'r') as f:
                data = json.load(f)
            
            for case in data:
                
                characteristics = (case['type'],)
                inv_pairs = 0
                total_pairs = 0
                
                try:
                
                    for rank, content in enumerate(case['sorted_reports']):
                        
                        name, value = content[0], content[1]
                        id = get_id(name)
                        
                        if os.environ.get('MIXED', '0') != '0':
                            if os.environ.get('IND_CLASSIFY', '0') == '1':
                                if not (4 <= content[0] < 6): continue
                            
                            if isinstance(content[0], int) and 0 <= content[0] < 6 and len(case['sorted_reports']) == 6:
                                (vals0 if content[0] % 2 == 0 else valsn0).append(content[1] / len(case['sorted_reports']))
                            else:
                                print(content[0], len(case['sorted_reports']), 'ERROR')
                        
                        else:
                            if id == 0:
                                vals0.append(value / len(case['sorted_reports']))
                            else:
                                valsn0.append(value / len(case['sorted_reports']))
                        
                        values_for_each[characteristics[0]][id].append(value)
                        positions_for_each[characteristics[0]][id].append(rank)
                        
                        for content2 in case['sorted_reports'][rank+1:]:
                            if not (id == 0 or get_id(content2[0]) == 0):
                                continue
                            
                            summation[characteristics] += int(id < get_id(content2[0]))
                            inv_pairs += int(id < get_id(content2[0]))
                            total_pairs += 1
                            expectation[characteristics] += 0.5
                
                except:
                    # print('Skipping case')
                    continue
                    
                entries[characteristics].append(inv_pairs)
                entries_null[characteristics].append(total_pairs * 0.5)
    
    else:
        data = []
        for file, _ in files:
            with open(file, 'r') as f:
                data += json.load(f)
            
            for case in data[-1]:
                # sort case['sorted_reports'] by the last field
                case['sorted_reports'].sort(key=lambda x: x[-1][0]+x[-1][1])
        
        min_len = min(len(d) for d in data)
        print('Min len:', min_len)
        
        for cases_tuple in zip(*data):
            
            characteristics = (cases_tuple[0]['type'],)
            inv_pairs = 0
            total_pairs = 0
            
            try:
            
                for case in cases_tuple:
                    assert case['question'] == cases_tuple[0]['question']
                
                all_reports = [case['sorted_reports'] for case in cases_tuple]
                
                for rank, all_contents in enumerate(zip(*all_reports)):
                    
                    name, value = all_contents[0][0], 0
                    id = get_id(name)
                    
                    for content in all_contents:
                        assert content[0] == name
                        assert content[-1][0] == all_contents[0][-1][0]
                        assert content[-1][1] == all_contents[0][-1][1]
                        value += content[1]
                    
                    if id == 0:
                        vals0.append(value / len(all_reports[0]))
                    else:
                        valsn0.append(value / len(all_reports[0]))
                    
                    values_for_each[characteristics[0]][id].append(value)
                    positions_for_each[characteristics[0]][id].append(rank)
                
            except:
                # print('Skipping case')
                continue

    with open('./raw_stats.json', 'w') as f:
        json.dump({'values_for_each': values_for_each, 'positions_for_each': positions_for_each}, f, indent=4)
        
    print(len(vals0), len(valsn0))
    X = np.array(vals0 + valsn0).reshape(-1, 1)
    y = np.array([0] * len(vals0) + [1] * len(valsn0))
    
    return get_full(get_score(X, y, silent=False), len(X), xs=X, ys=y)
    
    

def get_metrc(files, merge=False, use_counterfactual=True):
    if 'llmasajudge' in files[0][0]:
        return get_metrc_for_llmasajudge(files, merge)
    
    expectation = defaultdict(lambda: 0)
    summation = defaultdict(lambda: 0)
    entries = defaultdict(lambda: [])
    entries_null = defaultdict(lambda: [])
    values_for_each = defaultdict(lambda: [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []])
    positions_for_each = defaultdict(lambda: [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []])

    all_case_0_scores = []
    all_ref_case_0_scores = []
    
    if not merge:
        for file, ref_file in files:
            with open(file, 'r') as f:
                data = json.load(f)
            
            if ref_file is None:
                ref_file = file
            
            with open(ref_file, 'r') as f:
                ref_data = json.load(f)
            
            for case, ref_case in zip(data, ref_data):
                
                characteristics = (case['type'],)
                inv_pairs = 0
                total_pairs = 0
                
                try:
                
                    case_0_scores = []
                    ref_case_0_scores = []
                    
                    if not use_counterfactual:
                        for rank, content in enumerate(case['sorted_reports']):
                            if os.environ.get('IND_CLASSIFY', '0') == '1':
                                if '8b' not in content[0].lower(): continue
                            
                            if 'deceptive' in content[0].lower():
                                case_0_scores.append(content[1] / len(case['sorted_reports']))
                            else:
                                ref_case_0_scores.append(content[1] / len(case['sorted_reports']))
                    
                    else:
                        for rank, content in enumerate(case['sorted_reports']):
                            if get_id(content[0]) == 0:
                                case_0_scores.append(content[1] / len(case['sorted_reports']))
                        
                        for rank, content in enumerate(ref_case['sorted_reports']):
                            if get_id(content[0]) == 0:
                                ref_case_0_scores.append(content[1] / len(case['sorted_reports']))
                    
                        random.shuffle(ref_case_0_scores)
                        random.shuffle(case_0_scores)
                    
                    # for rank, content in enumerate(ref_case['sorted_reports']):
                    #     if get_id(content[0]) == 0:
                    #         ref_case_0_scores.append(content[1] / len(ref_case['sorted_reports']))
                    
                    if random.randint(0, 500) == 0: print(len(case['sorted_reports']), use_counterfactual, case_0_scores, ref_case_0_scores)
                    # assert len(case_0_scores) == len(ref_case_0_scores) == 1
                    
                    inv_pairs += int(case_0_scores[0] > ref_case_0_scores[0])
                    summation[characteristics] += int(case_0_scores[0] > ref_case_0_scores[0])
                    expectation[characteristics] += 0.5
                    total_pairs += 1
                    
                    if not use_counterfactual:
                        all_case_0_scores.extend(case_0_scores)
                        all_ref_case_0_scores.extend(ref_case_0_scores)
                    else:
                        all_case_0_scores.append(case_0_scores[0])
                        all_ref_case_0_scores.append(ref_case_0_scores[0])
                    
                    for rank, content in enumerate(case['sorted_reports']):
                        
                        name, value = content[0], content[1]
                        id = get_id(name)
                        
                        values_for_each[characteristics[0]][id].append(value / len(case['sorted_reports']))
                        positions_for_each[characteristics[0]][id].append(rank)
                
                except Exception as e:
                    # print('Skipping case', e)
                    # traceback.print_exc()
                    continue
                    
                entries[characteristics].append(inv_pairs)
                entries_null[characteristics].append(total_pairs * 0.5)
    
    else:
        for cur_files_id, cur_files in enumerate(zip(*files)):
            assert cur_files_id < 2
            
            data = []
            for file in cur_files:
                with open(file, 'r') as f:
                    data.append(json.load(f))
                
                for case in data[-1]:
                    # sort case['sorted_reports'] by the last field
                    case['sorted_reports'].sort(key=lambda x: x[-1][0]+x[-1][1])
            
            min_len = min(len(d) for d in data)
            print('Min len:', min_len)
            
            for cases_tuple in zip(*data):
                
                characteristics = (cases_tuple[0]['type'],)
                inv_pairs = 0
                total_pairs = 0
                
                try:
                
                    for case in cases_tuple:
                        assert case['question'] == cases_tuple[0]['question']
                    
                    all_reports = [case['sorted_reports'] for case in cases_tuple]
                    
                    for rank, all_contents in enumerate(zip(*all_reports)):
                        
                        name, value = all_contents[0][0], 0
                        id = get_id(name)
                        if id and cur_files_id == 0:
                            continue
                        
                        for content in all_contents:
                            assert content[0] == name
                            assert content[-1][0] == all_contents[0][-1][0]
                            assert content[-1][1] == all_contents[0][-1][1]
                            value += content[1]
                        
                        if cur_files_id == 0:
                            all_case_0_scores.append(value / len(all_reports[0]))
                        else:
                            all_ref_case_0_scores.append(value / len(all_reports[0]))
                        
                        values_for_each[characteristics[0]][id].append(value)
                        positions_for_each[characteristics[0]][id].append(rank)
                    
                except:
                    # print('Skipping case')
                    continue
    
    print('Data volume:', len(all_case_0_scores), len(all_ref_case_0_scores))
    
    with open('./raw_stats.json', 'w') as f:
        json.dump({'values_for_each': values_for_each, 'positions_for_each': positions_for_each}, f, indent=4)

    print('Counterfactual Test Results:')
    mean = np.mean(all_case_0_scores)
    mean_ref = np.mean(all_ref_case_0_scores)
    print(f"Means = {mean:.2f} vs {mean_ref:.2f}")
    # print(f"P-Value {stats.ttest_rel(all_case_0_scores, all_ref_case_0_scores, alternative='less')[1]}")

    print(len(all_case_0_scores), len(all_ref_case_0_scores))
    X = np.array(all_case_0_scores + all_ref_case_0_scores).reshape(-1, 1)
    y = np.array([0] * len(all_case_0_scores) + [1] * len(all_ref_case_0_scores))

    return get_full(get_score(X, y, silent=False), len(X), xs=X, ys=y)


if __name__ == '__main__':
    # print(get_metrc([
    #     ('./results/20240912-witness-both-dedup-nowild-n4/llmasajudge-qwen2.5-7B-1.json', None),
    # ]))
    
    print(get_metrc([
        ('./results/20240912-witness-both-dedup-nowild-n4-mixed/sorted_reports-1-4.json', None),
    ], merge=False, use_counterfactual=False))
    
    # print(get_metrc([
    #     ('./results/20240912-witness-both-dedup-nowild-n4/sorted_reports-1-4-smol-360M-1.json', './results/20240912-witness-both-dedup-nowild-n4/sorted_reports-0-2-smol-360M-0.json'),
    # ]))