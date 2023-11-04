from evaluate.visualization import radar_plot
from models.predict_model import *


data = []
for model_name in MODEL_NAMES:
    # Opening JSON file
    with open(f'../references/{model_name}.json') as json_file:
        test_results = json.load(json_file)
        data.append({"BLEU": test_results[model_name]['score'],
                     "Average inference time": test_results[model_name]['average_inf_time'],
                     "Brevity penalty: ": test_results[model_name]['bp']
                     })
plot = radar_plot(data=data, model_names=MODEL_NAMES)
plot.show()
