# Automated Classification of Pork Rasher Packaging Errors

**MMA3001 Numerical Methods and Machine Learning: Project** Yuwan Suriyakumara (ID: 34953302)


A computational vision decision support tool, created for packaging quality control. Given an image of a packaged pork rasher tray, it concludes a pass or fail, for failed trays it undergoes a further extension to classify the defect type (loose-meant, packaging error, twisted meat, unsealed, wrinkles). This has been designed for a human in the loop setting, flagged trays will to an inspector, hence the system will never accept or reject a product on its own.  

The method used is transfer learning (a frozen MobileNetV2 backbone with a small trainable head) on the Roboflow
*Pork Rasher Error (Packaging)* dataset. Bounding box annotations are converted to image level lavels so the project performs classfications rather than object detection. The full method, valdiation, alongside limitations are written in the report. 

## Results 
All results are from a single training run per model and are evaluated on the dataset's test split.

| Task | Result |
|---|---|
| Pass/fail, default threshold 0.5 | 80% accuracy; 13 of 68 real defects missed |
| Pass/fail, selected threshold 0.3 | 67 of 68 defects caught, but 76 of 80 trays (95%) are still sent for review, so labour saving on this test set is small |
| From-scratch CNN (comparison) | Collapsed to always predicting "fail" (0 of 12 pass images correct), despite 81x fewer parameters |
| Defect type (5 classes) | 72% accuracy, only slightly above the 70.8% of always predicting "unsealed"; macro-F1 0.27. Reliable only for `unsealed` (F1 0.92) |

**Important limitations** (detailed in the report): the decision threshold was chosen on the test set, so the metrics at 0.3 are considerably optimistic, further wrinkle has no validation or test examples so its metrics are undefined. The test set is about 85% defective, which in reality, a real line would not be. The interference time was not measured and the two models were not evaluated end to end. 


## Repository structure

```
.
├── src/
│   ├── data_utils.py        # label conversion from YOLO boxes + tf.data pipelines
│   └── model_utils.py       # models, class weights, training, evaluation
├── tests/
│   ├── conftest.py          # synthetic Roboflow-style dataset used by the tests
│   ├── test_data_utils.py
│   └── test_model_utils.py
├── docs/
│   ├── html/                # pdoc API documentation (open html/index.html)
│   ├── test_report.html     # pytest report
│   └── coverage_summary.txt # test coverage summary
├── PROJECT.ipynb            # end-to-end workflow notebook (run in Google Colab)
├── requirements.txt         # runtime dependencies
├── requirements-dev.txt     # + testing and documentation tools
├── pytest.ini
├── LICENSE
└── README.md
```

## Setup and usage

The project is designed to run in **Google Colab** (free GPU tier), where TensorFlow, pandas, Pillow and
scikit-learn are preinstalled.

1. Open `PROJECT.ipynb` in Colab and set *Runtime → Change runtime type → GPU (T4)*.
2. Get a free [Roboflow](https://roboflow.com) API key and store it as a Colab **Secret** named `ROBOFLOW_API_KEY`
3. Choose *Runtime → Run all* (roughly 10 to 15 minutes). The notebook clones this repository, downloads the dataset,
   reproduces the data audit, trains the baseline, the from-scratch CNN and the defect-type model, and prints the tables
   and figure used in the report.

Results vary slightly between runs because random seeds are not fixed.

## Testing

```bash
python -m pytest                                                       # run the suite
python -m pytest --html=docs/test_report.html --self-contained-html --cov=src   # regenerate the report
```

The suite has **45 tests** and **97% line coverage** of `src/` (see `docs/coverage_summary.txt`). It uses a small
synthetic dataset built on the fly, so it runs offline in about 15 seconds without the real dataset or a GPU.
It covers the label-derivation policy (corrupt, unlabelled, repeated-box and multi-defect images), class weights against
hand calculation, the parameter counts quoted in the report, the effect of the decision threshold, and the zero-example
`wrinkle` case. The few uncovered lines are the body of a function that runs inside TensorFlow's compiled graph, which
the coverage tool cannot observe.

## Documentation

API documentation is generated from the Google-style docstrings with [pdoc](https://pdoc.dev):

```bash
pdoc -d google -o docs/html src/data_utils.py src/model_utils.py
```

## Licences

The code is released under the [MIT License](./LICENSE).

The dataset is used under its own licence (CC BY 4.0) and is **not** redistributed in this repository.

## Citation

Hello. (2025). *Pork Rasher Error (Packaging)* [Data set]. Roboflow Universe.
https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging (CC BY 4.0)

## AI Acknowledgement

This project was developed with the assistance of an AI tool, in line with the unit's requirements. Full detail and
reflection are in Section 8 of the written report.

**Tool:** Claude (Anthropic, model Claude Sonnet 5.5), used via claude.ai from 15/09/2026 to 22/10/2026 No other AI tool was used.

**What it was used for:**
- Generating the source code (`data_utils.py`, `model_utils.py`), the pytest suite, the pdoc documentation, this README
  and the notebook. I ran, tested and debugged the code in Google Colab.
- Interpreting training and evaluation output, and designing the threshold and class-weighting analyses.
- Drafting the written report: the AI wrote first drafts of most sections and tables from my own results, which I
  then edited, corrected and checked.

**Level of contribution:** high for first drafts of the code, tests and report text, and moderate for interpreting
results. The project decisions, the running of training, evaluation and the data audit, and every result reported come from me.
The tests, deliberate-bug check and HTML documentation were first run by the AI in its own environment, and I re-ran them in Colab.

**How it was checked:** all code was executed in my own environment and errors were found and fixed by running it;
every reported number comes from my own output; derived figures were cross-checked. A final check of the AI-drafted
report found and corrected several errors, which are listed in the report.

**Decisions that remained mine:** the dataset and scope (classification rather than detection); keeping the `wrinkle`
class against the AI's initial suggestion to drop it; the decision threshold; and verifying the results.

I remain responsible for all code, results and claims in this repository.
