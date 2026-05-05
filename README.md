# Path2Win: Interpretable Pregame Win Prediction for Professional League of Legends

This repository contains the code, data, and output summaries for **Path2Win**, an interpretable pregame win-probability model for professional League of Legends matches from the 2024 LPL Summer Season.

The goal of this project is to predict, before a match begins, the probability that the **blue side wins**, using only information available before the match. In addition to predicting the outcome, Path2Win is designed to explain the predicted route to victory through interpretable win-condition indices.

---

## Project objective

Professional League of Legends matches depend on many interacting factors, including lane pressure, role-specific player strength, objective control, teamfighting, jungle resource control, and patch-specific game dynamics.

Rather than building only a black-box classifier, this project formulates pregame prediction as an interpretable pathway problem:

```text
Pregame information
→ predicted win-condition advantages
→ interpretable condition indices
→ blue-side win probability
