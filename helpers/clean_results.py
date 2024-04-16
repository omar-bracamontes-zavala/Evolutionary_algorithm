import json

# Opening JSON file
mu_f = open('results_mu.json')
mu_data = json.load(mu_f)

results_f = open('results.json')
data = json.load(results_f)


selection_shift = 1
replacement_shift = 2

for design in mu_data:
    # Selection functions are in col 0
    design['combination_indexes'][0] += selection_shift
    # Replacement functions are in col -1
    design['combination_indexes'][-1] += replacement_shift
    
print(json.dumps(mu_data,indent=4))

print(len(data), len(mu_data))


new_data = data+mu_data

print(len(new_data))

with open('updated_results.json','w') as json_file:
    json.dump(new_data,json_file)