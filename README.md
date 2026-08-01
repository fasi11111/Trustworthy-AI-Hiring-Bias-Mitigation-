# IS4T702 MSc Project: Trustworthy AI
## A Comparative Experimental Study of Bias Mitigation Techniques in AI Driven Hiring Algorithms

**Student:** Faisal Aleem  
**Supervisor:** Janusz  
**University:** University of South Wales  
**Programme:** MSc Computer Science with Artificial Intelligence  
**Year:** 2025/2026  

---

## Project Overview

This project investigates and compares four bias mitigation techniques applied to AI driven hiring algorithms. The study evaluates techniques from all three mitigation families (preprocessing, in processing, postprocessing) across two benchmark datasets and multiple fairness metrics.

**Datasets:**
- UCI Adult Income Dataset (protected attribute: sex)
- COMPAS Recidivism Dataset (protected attribute: race)

**Techniques compared:**
1. Baseline (no mitigation)
2. Reweighing (Kamiran and Calders, 2012)
3. Disparate Impact Remover (Feldman et al., 2015)
4. Prejudice Remover (Kamishima et al., 2012)
5. Equalised Odds Postprocessing (Hardt et al., 2016)

**Fairness metrics:**
- Demographic Parity Difference (DPD)
- Average Odds Difference (AOD)
- Equalised Odds Difference (EOD)
- Disparate Impact Ratio (DIR)

---

## Repository Structure

```
project/
   prototype.py          Main experimental pipeline script
   requirements.txt      Python package dependencies
   results_adult.csv     Experimental results: UCI Adult dataset
   results_compas.csv    Experimental results: COMPAS dataset
   results_chart.png     Comparison bar charts (all metrics, both datasets)
   tradeoff_chart.png    Accuracy vs. fairness trade-off plot
   README.md             This file
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Experiment

```bash
python prototype.py
```

### Expected output:
- A comparison table printed to the terminal
- bias_mitigation_results.csv saved to the current directory
- bias_mitigation_chart.png saved to the current directory

---

## Dataset Setup

The UCI Adult dataset is bundled with AIF360. If it is not found automatically, download the following files and place them in your AIF360 data directory:

```
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
```

Place them in: `<aif360_install_path>/data/raw/adult/`

The COMPAS dataset can be downloaded from:
```
https://github.com/propublica/compas-analysis
```

---

## Key Results (Preliminary)

| Technique             | Accuracy | DPD Improvement | Best For        |
|-----------------------|----------|-----------------|-----------------|
| Baseline              | 0.8267   | 0%              | Reference only  |
| Reweighing            | 0.8083   | 78%             | Balanced use    |
| Dispar. Impact Rem.   | 0.7992   | 95%             | Demographic parity |
| Prejudice Remover     | 0.7937   | 83%             | Consistent performance |
| Equalised Odds PP     | 0.7891   | 62% (DPD)       | Equalised odds  |

---

## References

- Bellamy et al. (2019) AI Fairness 360. IBM Journal of Research and Development.
- Bird et al. (2020) Fairlearn. Microsoft Research.
- Feldman et al. (2015) Certifying and removing disparate impact. KDD 2015.
- Hardt et al. (2016) Equality of opportunity in supervised learning. NeurIPS 2016.
- Kamiran and Calders (2012) Data preprocessing techniques. Knowledge and Information Systems.
- Kamishima et al. (2012) Fairness-aware classifier. ECML PKDD 2012.
- Pedregosa et al. (2011) Scikit-learn. Journal of Machine Learning Research.

---

*For questions, contact: Faisal Aleem | University of South Wales*
