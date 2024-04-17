import json

def filter_top_n_by_metric(data, n, metric):
    # Sort the data by the 'MBF' value, using a lambda function to access the nested 'MBF'
    sorted_data = sorted(data, key=lambda x:x['performance_metrics'][metric])
    # Return the top n entries
    return sorted_data[:n]

def get_top_summary(top_n, metric):
    top_designs = list(
        map(
            lambda x: (x['combination_indexes'], x['performance_metrics'][metric]),
            top_n
        )
    )
    
    print(f'Top {len(top_n)} by {metric}:')
    for i, design in enumerate(top_designs):
        print(f"\t{i+1}. {design}")
        
    return [design for design,_ in top_designs]

def get_top_n_by_metrics(data, n=3):
    top_n_designs_bt_mbf = filter_top_n_by_metric(data, n, 'MBF')
    top_mbf = get_top_summary(top_n_designs_bt_mbf, 'MBF')

    top_n_designs_bt_pp = filter_top_n_by_metric(data, n, 'peak_performance')
    top_pp = get_top_summary(top_n_designs_bt_pp, 'peak_performance')

    return top_mbf, top_pp

def get_merged_tops(data_filepath, n=3):
    results_f = open(data_filepath)
    data = json.load(results_f)

    top_mbf, top_pp = get_top_n_by_metrics(data, n)

    # Convert all inner lists to tuples for both list1 and list2
    set1 = set(tuple(item) for item in top_mbf)
    set2 = set(tuple(item) for item in top_pp)

    # Union the sets to merge and remove duplicates
    merged_set = set1.union(set2)

    # Convert the tuples back to lists
    merged_list = [list(item) for item in merged_set]

    print(merged_list)
    return merged_list



top_designs = get_merged_tops('results/cleaned_results.json', 3)