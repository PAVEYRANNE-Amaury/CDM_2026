# ⚽ World Cup 2026 Simulation Model

This project is a probabilistic simulation framework for predicting football match outcomes and estimating World Cup 2026 winning probabilities using a Poisson-based generative model.

⚠️ **Work in progress — project for fun**

---

## 🧠 Overview

The goal of this project is to build a full pipeline that:

- learns team strength parameters from historical international matches
- models goals using a Poisson goal-generation process
- simulates full tournaments (group stage + knockout phase)
- estimates winning probabilities via Monte Carlo simulation

The model is inspired by classical statistical approaches to football modeling (Poisson attack/defense models).

---

## 📊 Methodology

### 1. Data
Historical international matches are used to estimate team strength:

- home/away scores
- match dates
- team identities

Data is periodically updated.

---

### 2. Model

Each team is assigned:

- an **attack strength**
- a **defensive strength**

Expected goals are modeled as:

- λ_home = exp(attack_home + defense_away)
- λ_away = exp(attack_away + defense_home)

Goals are then sampled using a Poisson distribution.

---

### 3. Group Stage Simulation

- Full round-robin simulation inside each group
- Real matches (if available) are integrated directly
- Remaining matches are simulated
- Ranking based on:
  - points
  - goal difference
  - goals scored

---

### 4. Knockout Phase

- Round of 32 constructed using FIFA-like rules
- Includes best third-placed teams
- Single-elimination format
- Draw resolution via extra stochastic goal sampling

---

### 5. Monte Carlo Simulation

The full tournament is simulated N times to estimate:

- probability of winning the World Cup
- distribution of finalists
- team performance variability

---

## 🚀 How to run

### Install dependencies

```bash
pip install -r requirements.txt
