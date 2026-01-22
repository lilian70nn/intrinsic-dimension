import numpy as np

def merged_over_categories(results):

    """
    Average ID measures across multiple categories.

    """

    merged_result = {}
    model_names = results[0].keys()
    for model_name in model_names:
        depths = results[0][model_name]["depths"]
        twonn_per_cat = [res[model_name]["TwoNN"] for res in results]
        mle_per_cat = [res[model_name]["MLE"] for res in results]


        twonn_avg = np.array(twonn_per_cat, dtype=float).mean(axis=0).tolist()
        mle_avg = np.array(mle_per_cat, dtype=float).mean(axis=0).tolist()

        merged_result[model_name] = {
            "depths": depths,
            "TwoNN": twonn_avg,
            "MLE": mle_avg
        }

    return merged_result