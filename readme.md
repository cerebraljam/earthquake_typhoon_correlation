
# Are Typhoon Causing Earthquakes? Testing between 2 hypotheses

tl;dr: It's unlikely

Saturday October 12th 2019, the Super Typhoon 19 has hit Japan. While chatting online with my friends, I asked if it was more likely to have earthquakes during typhoons than it is normally. One of my friend told me that 

> On theory is that there’s actually a confounding going on. Both earthquakes, typhoons, lightning storms being in part driven by cosmic radiation. According to the theory such radiation gets amplified at certain time of year, roughly March and September. So, the impact to earths magnetic field is larger then. 3-11 Tohoku earthquake was in March. Could be coincidence but take a look at the timings of large earthquakes and you will see some bias toward these times of year. 

> Refering to [Australian Geographic](https://www.australiangeographic.com.au/topics/science-environment/2011/03/earthquakes-the-10-biggest-in-history/), out of the 10 largest, 3 are in March and one in end of Feb. 4/12 could be by chance of course.

On this, I decided to test both hypotheses:
* Earthquakes could be driven by cosmic radiations, roughly around March and September
* There is a higher chance of earthquakes during typhoons.

While toying around with the first hypothesis: 18:21:53 JPT, Chiba was hit by a M5.7 Shindo 4 earthquake, which strangely privided evidences for my initial hypothesis, but was it really?

## Hypothesis 1: Earthquakes could be driven by cosmic radiations, roughly around March and September

Using the data from Australian Geographic, we can check if it is significant:

Hypothesis 1: There are more major earthquakes in March and September. The sample is quite limited (5, 3, 12, 3, 11, 2, 1, 4, 3, 8). Still, we can try anyway. 

Hypothesis 2: The opposing hypothesis is that there is no relation and that earthquakes are relatively random

To test this, I will oppose the odds of both hypotheses. Earthquakes happening more often in March and September is the equivalent of throwing loaded dice: 
* h1 (`(1/6)^4 + (5/6)^6`) should be much greater than h2 (`(1/12)^4 + (11/12)^6)`) if cosmic radiations have an impact.
* Otherwise, the ratios should be quite close

How does this calculation works? For more details, have a look at Will Kurt's book [Bayesian Statistics the Fun Way](https://www.amazon.co.jp/Bayesian-Statistics-Fun-Will-Kurt/dp/1593279566/ref=sr_1_1?keywords=bayesian+the+fun+way&qid=1571041389&sr=8-1), or his blog article [Bayesian Reasoning in The Twilight Zone!](https://www.countbayesie.com/blog/2016/3/16/bayesian-reasoning-in-the-twilight-zone)

If this I am using Bayesian Factors, where are the priors? Well, hoping to find something, I purposely left myself easy to convince by using prior odds of 1:2. I was honestly hoping to be able to crank this up later if results were favorable.


```python
import numpy as np
from scipy.stats import beta
import pandas as pd
import datetime


# We start by counting the values of Alpha and Beta:
# Alpha being the number of times the earthquake values hits March or September
# Beta is when it fails to match
def evaluate_alphabeta(hits, values, a, b):
    for m in values:
        if m in hits:
            a+=1
        else:
            b+=1
    return a, b

# Then we compare against
def evaluate_odds(hit, value, a, b, comment=True):
    # P(D|h1) 
    x = beta.stats(hit, value - hit, moments='m') # probability of hits among the sample
    h1 = np.power(x, a) * np.power(1 - x, b)

    # P(D|h2) 
    y = beta.stats(1, value - 1, moments='m') # probability that it is random (1 / 12)
    h2 = np.power(y, a) * np.power(1 - y, b)
    
    # P(D|H1) / P(D|H2)
    o = h1/h2
    if comment:
        print('* Probability of having earthquakes more often: {}'.format(h1))
        print('* Probability that earthquakes are uniformly distributed: {}'.format(h2))
        print('* Probability that H1 explains more observed data than H2: {}'.format(o))
        
    return o

def comment_odds(o):
    if o < 1:
        return 'Nothing interesting'
    if 1 <= o and o < 3:
        return 'Interesting, but nothing conclusive'
    elif 3 <= o and o < 20:
        return "Looks like we're on something"
    elif 20 <= o and o < 150:
        return "Strong evidence in favor of H1"
    elif 150 <= o:
        return "Overwhelming evidence"

# months of earthquakes from https://www.australiangeographic.com.au/topics/science-environment/2011/03/earthquakes-the-10-biggest-in-history/
majors = [5, 3, 12, 3, 11, 2, 1, 4, 3, 8]
valids = [x+1 for x in range(12)]

hits = [3, 9] # the hypothesis is that earthquakes happens more often in March and September

# prior odds of 1 success for 2 failure
prior_a = 1 # alpha
prior_b = 2 # beta

a, b = evaluate_alphabeta(hits, majors, prior_a, prior_b)
print('alpha: {}, beta: {}'.format(a, b))
o = evaluate_odds(2, 12, a, b)
print("  * {}".format(comment_odds(o)))
```

    alpha: 4, beta: 9
    * Probability of having earthquakes more often: 0.0001495422063794868
    * Probability that earthquakes are uniformly distributed: 2.203829376497226e-05
    * Probability that H1 explains more observed data than H2: 6.7855618939597635
      * Looks like we're on something


It seems that h1 slightly better explain the observations than h2. But how about if we try with more earthquakes?

## Testing with more data
For this next test, I used the [earthquake dataset available on Kaggle](https://www.kaggle.com/usgs/earthquake-database).

First I need to adjust the data to suite the checks, by converting the date in string into a datetime format, then adding a column Month that contains only the month number.


```python
df_eq = pd.read_csv('earthquake-database.csv')
df_eq['Month'] = 0

for i, r in df_eq.iterrows():
    s = r['Date']
    try:
        d = datetime.datetime.strptime(s, '%m/%d/%Y')
    except:
        d = datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%fZ')
    df_eq.at[i, 'Date'] = d
    df_eq.at[i, 'Month'] = d.month

print(len(df_eq))
```

    23412


Next, we just replace the data from Australian Geographic with the values from the Dataset from Kaggle.


```python
magnitude=7.2

majors = list(df_eq[df_eq['Magnitude'] >= magnitude]['Month'])
valids = [x+1 for x in range(12)]

hits = [3, 9] # the hypothesis is that earthquakes happens more often in March and September

# prior odds of 1 success for 2 failure
prior_a = 1 # alpha
prior_b = 2 # beta

a, b = evaluate_alphabeta(hits, majors, prior_a, prior_b)
print('alpha: {}, beta: {}'.format(a, b))
k1 = evaluate_odds(2, 12, a, b)
print("  * {}".format(comment_odds(k1)))
```

    alpha: 69, beta: 381
    * Probability of having earthquakes more often: 1.3788244609655893e-84
    * Probability that earthquakes are uniformly distributed: 1.3773754052558502e-89
    * Probability that H1 explains more observed data than H2: 100105.2041225805
      * Overwhelming evidence


Overwhelming evidences? Really? What if we try with other months?


```python
majors = list(df_eq[df_eq['Magnitude'] >= magnitude]['Month'])
hits = [1, 6] # the hypothesis is that earthquakes happens more often in March and September

# prior odds of 1 success for 2 failure
prior_a = 1 # alpha
prior_b = 2 # beta

a, b = evaluate_alphabeta(hits, majors, prior_a, prior_b)
print('alpha: {}, beta: {}'.format(a, b))
k2 = evaluate_odds(2, 12, a, b)
print("  * {}".format(comment_odds(k2)))

```

    alpha: 63, beta: 387
    * Probability of having earthquakes more often: 2.1544132202587347e-80
    * Probability that earthquakes are uniformly distributed: 2.4401045503104597e-83
    * Probability that H1 explains more observed data than H2: 882.9184060923226
      * Overwhelming evidence


Oh oh, January and June also provide overwhelming evidences even better suite the data we have. This tells us that we can't simply blindly say "Yeah, Cosmic Radiations, mainly happening in March and September, causes earthquakes". We need more tests.


```python
print("k1 vs k2 ({}): {}".format(k1/k2, comment_odds(k1/k2)))
print("k2 vs k1 ({}): {}".format(k2/k1, comment_odds(k2/k1)))
```

    k1 vs k2 (113.37990399999994): Strong evidence in favor of H1
    k2 vs k1 (0.008819905157090276): Nothing interesting


Hummm, k1 (Hypothesis 1 with kaggle data) seems far superior over k2. How about the other months? Testing all of them by hand would be quite bothering. 

## Finding the best hypotheses using Monte Carlo Simulations

The cell below is probably way too complicated for what it needs to be, but because of the large quantity of earthquakes in the Kaggle database, running the `evaluate_odds` function on all will case overflows. To go around this, I run 1000 simulations of 400 samples for each pairs of months, then I make an average of the results to evaluate the more likely value. Since there are only 450 earthquakes of 7.2 and higher, it should tend to the average relatively quickly. Still, we can later run this simulation on smaller earthquakes as well.


```python
def monte_carlo_simulation(simulation, sample, prior_a, prior_b):
    outcome = {}
    print('Checking our hypothesis against {} simulations of {} random samples of earthquakes >= {} ({})'.format(simulation, sample, magnitude, len(list(df_eq[df_eq['Magnitude'] >= magnitude]['Month']))))
    print('My priors are that there is a {} chance against {} that there is a relation between earthquakes and months'.format(prior_a, prior_b))
    for x in range(1, 12):
        for y in range(x+1, 13):
            candidates = [x, y]
            xy = ",".join([str(x), str(y)])

            for i in range(simulation):
                earthquakes = np.random.choice(list(df_eq[df_eq['Magnitude'] >= magnitude]['Month']), sample)

                a, b = evaluate_alphabeta(candidates, earthquakes, prior_a, prior_b)

                o = evaluate_odds(2, 12, a, b, False)

                if xy not in outcome.keys():
                    outcome[xy] = np.empty(0)

                outcome[xy] = np.append(outcome[xy], o)
    report = {}
    for xy in outcome.keys():
        if outcome[xy].mean() not in report.keys():
            report[outcome[xy].mean()] = []
        report[outcome[xy].mean()].append(xy)

    print("Average of the odds from all the pair of months: {}: {}".format(np.array(list(report.keys())).mean(), comment_odds(np.array(list(report.keys())).mean())))
    for s in sorted(report.keys(), reverse=True)[:10]:
        print("{}: {}: {}".format(s, report[s], comment_odds(s)))
```


```python
simulation = 1000
sample = 400

prior_a, prior_b = 1, 2
monte_carlo_simulation(simulation, sample, prior_a, prior_b)
```

    Checking our hypothesis against 1000 simulations of 400 random samples of earthquakes >= 7.2 (447)
    My priors are that there is a 1 chance against 2 that there is a relation between earthquakes and months
    Average of the odds from all the pair of months: 6.111660292559432e+16: Overwhelming evidence
    2.1884519958117307e+18: ['7,12']: Overwhelming evidence
    1.0047165259206203e+18: ['8,12']: Overwhelming evidence
    6.043539757084776e+17: ['7,11']: Overwhelming evidence
    9.951259896483709e+16: ['7,8']: Overwhelming evidence
    7.771297634801378e+16: ['8,10']: Overwhelming evidence
    3.657458943097335e+16: ['8,11']: Overwhelming evidence
    6054203548371027.0: ['11,12']: Overwhelming evidence
    4127652739164493.0: ['1,8']: Overwhelming evidence
    3560460980821601.5: ['10,11']: Overwhelming evidence
    1914229552051700.5: ['10,12']: Overwhelming evidence


Now they are all `Overwhelming evidence`, even the average. This tells us one thing useful: if the sample size is too great, this method of comparing odds degenerates. 

Running the simulation on a smaller sample gives more reasonable results, but it doesn't change the fact that all results are still quite similar, and not singling out one hypothesis.


```python
simulation = 1000
sample = 10

prior_a, prior_b = 1, 2
monte_carlo_simulation(simulation, sample, prior_a, prior_b)
```

    Checking our hypothesis against 1000 simulations of 10 random samples of earthquakes >= 7.2 (447)
    My priors are that there is a 1 chance against 2 that there is a relation between earthquakes and months
    Average of the odds from all the pair of months: 3.9811087970364296: Looks like we're on something
    5.917129862368336: ['8,10']: Looks like we're on something
    5.435147805865132: ['8,12']: Looks like we're on something
    5.26815845365071: ['7,8']: Looks like we're on something
    5.256732085387539: ['7,11']: Looks like we're on something
    4.977336642310346: ['7,12']: Looks like we're on something
    4.950557744393354: ['4,8']: Looks like we're on something
    4.893458155653582: ['8,11']: Looks like we're on something
    4.84951487722067: ['10,11']: Looks like we're on something
    4.755355708194566: ['3,8']: Looks like we're on something
    4.739861795776273: ['10,12']: Looks like we're on something


From the data available, it doesn't seems like Cosmic Radiations have a significant impact on earthquakes.

## Hypothesis 2: There is a higher chance of earthquakes during typhoons.

The second hypothesis was that Typhoon could cause earthquakes. To test this hypothesis, I used the [Hurricaine dataset from Kaggle](https://www.kaggle.com/noaa/hurricane-database). 

Since there are more earthquakes in the Pasific than the Atlantic ocean, I limited the longitude between Bangkok and the Marshall Islands. This is arbitray, but considering the size of typhoons and their strenght, it could be arguable that their effect could extend on long distances.

First, we need to load the dataset, fix the dates, longitudes and latitudes to match the Earthquake dataset.


```python
df_typhoon = pd.read_csv('typhoon-pacific.csv')

df_typhoon['Month'] = 0
longitude_min = 129
longitude_max = 163

for i, r in df_typhoon.iterrows():
    s = str(r['Date'])
    d = datetime.datetime.strptime(s, '%Y%m%d')
    df_typhoon.at[i, 'Date'] = d
    
    la = r['Latitude']
    letter = la[len(la)-1:]
    la_coordinate = float(la[:len(la)-1])
    if letter == 'S':
        la_coordinate*= -1
    
    lo = r['Longitude']
    letter = lo[len(lo)-1:]
    lo_coordinate = float(lo[:len(lo)-1])
    
    if letter == 'E':
        lo_coordinate*= -1
    
    df_typhoon.at[i, 'Latitude'] = float(la_coordinate)
    df_typhoon.at[i, 'Longitude'] = float(lo_coordinate)

# We need to limit the typhoon and earthquake dataset to their overlapping periods
df_typhoon_period = df_typhoon[(df_typhoon_period['Longitude'] >= 129) & (df_typhoon_period['Longitude'] <= 163) & (df_typhoon['Date'] >= min(df_eq['Date']))]
df_eq_period = df_eq[(df_eq_period['Longitude'] >= 129) & (df_eq_period['Longitude'] <= 163) & (df_eq['Date'] <= max(df_typhoon['Date']))]
```

With the data prepared, I can prepare the values to run the same tests as for the earthquakes and cosmic radiations.

* condition for a: There is an earthquake the same day as a typhoon
* condition for b: There is a typhoon without an earthquake


```python
hits = df_eq_period[(df_eq_period['Magnitude'] >= magnitude)]['Date'].unique()
valids = df_typhoon_period['Date'].unique()

# prior odds of 1 success for 2 failure
prior_a = 1 # alpha
prior_b = 2 # beta

a, b = evaluate_alphabeta(hits, valids, prior_a, prior_b)
print('alpha: {}, beta: {}'.format(a, b))
print('earthquakes during the period: {}, typhoon during the same period: {}, a: {}, b: {}'.format(len(hits), len(valids), a, b))
k3 = evaluate_odds(len(hits), len(valids), a, b, False)
print("  * {}".format(comment_odds(k3)))
```

    alpha: 8, beta: 1330
    earthquakes during the period: 111, typhoon during the same period: 1335, a: 8, b: 1330
      * Nothing interesting


Well, from the datasets we have, longitude restrictions, and the earthquake size I chose, it seems that there isn't any correlation between typhoon and major earthquakes. 

The storms might not be strong enough to cause tectonic plates to move.
