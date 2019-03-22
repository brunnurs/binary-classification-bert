import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def predict(original_test_data, test_data_loader, model, device):

    input_data = [{'id': test_example.guid,
                   'sentence': test_example.text_a,
                   'true_label': test_example.label} for test_example in original_test_data]

    all_logits = None

    model.eval()
    nb_eval_steps, nb_eval_examples = 0, 0
    for step, batch in enumerate(tqdm(test_data_loader, desc="Prediction Iteration")):
        input_ids, input_mask, segment_ids, _ = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            logits = logits.sigmoid()

        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    return pd.merge(pd.DataFrame(input_data), pd.DataFrame(all_logits, columns='predicted_label'))
