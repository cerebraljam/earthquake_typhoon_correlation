
# Are Typhoon Causing Earthquakes? Testing between 2 hypotheses

tl;dr: I still have doubts, but... it does seem so?

Saturday October 12th 2019, the Super Typhoon 19 has hit Japan. While chatting online with my friends, I asked if it was more likely to have earthquakes during typhoons than it is normally. One of my friend told me that 

> On theory is that there’s actually a confounding going on. Both earthquakes, typhoons, lightning storms being in part driven by cosmic radiation. According to the theory such radiation gets amplified at certain time of year, roughly March and September. So, the impact to earths magnetic field is larger then. 3-11 Tohoku earthquake was in March. Could be coincidence but take a look at the timings of large earthquakes and you will see some bias toward these times of year. 

> Refering to [Australian Geographic](https://www.australiangeographic.com.au/topics/science-environment/2011/03/earthquakes-the-10-biggest-in-history/), out of the 10 largest, 3 are in March and one in end of Feb. 4/12 could be by chance of course.

On this, I decided to test both hypotheses:
* Earthquakes could be driven by cosmic radiations, roughly around March and September
* There is a higher chance of earthquakes during typhoons.

While toying around with the first hypothesis: 18:21:53 JPT, Chiba was hit by a M5.7 Shindo 4 earthquake, which strangely privided evidences for my initial hypothesis, but was it really?

## Hypothesis 1: Earthquakes could be driven by cosmic radiations, roughly around March and September

Using the data from Australian Geographic, we can check if it is significant:

* Hypothesis 1: There are more major earthquakes in March and September. The sample is quite limited (5, 3, 12, 3, 11, 2, 1, 4, 3, 8). Still, we can try anyway. 

* Hypothesis 2: The opposing hypothesis is that there is no relation and that earthquakes are relatively random

To test this, I will oppose the odds of both hypotheses. Earthquakes happening more often in March and September is the equivalent of throwing loaded dice: 
* h1 (`(1/6)^4 + (5/6)^6`) should be much greater than h2 (`(1/12)^4 + (11/12)^6)`) if cosmic radiations have an impact.
* Otherwise, the ratios should be quite close

How does this calculation works? For more details, have a look at Will Kurt's book [Bayesian Statistics the Fun Way](https://www.amazon.co.jp/Bayesian-Statistics-Fun-Will-Kurt/dp/1593279566/ref=sr_1_1?keywords=bayesian+the+fun+way&qid=1571041389&sr=8-1), or his blog article [Bayesian Reasoning in The Twilight Zone!](https://www.countbayesie.com/blog/2016/3/16/bayesian-reasoning-in-the-twilight-zone)

If this I am using Bayesian Factors, where are the priors? Any prior would have done, but let's be skeptical: 1:10 seems reasonable.


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
#     print(hit, value - hit, x, h1)

    # P(D|h2) 
    y = beta.stats(1, value - 1, moments='m') # probability that it is random (1 / 12)
    
    h2 = np.power(y, a) * np.power(1 - y, b)
#     print(1, value - 1, y, h2)
    
    # P(D|H1) / P(D|H2)
    o = h1/h2
    if comment:
        print('* Probability of h1 with alpha {} and beta {}: {}'.format(a, b, h1))
        print('* Probability of having all the events uniforally distributed with the same alpha/beta: {}'.format(h2))
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
prior_a = 1 # prior alpha
prior_b = 10 # prior beta

a, b = evaluate_alphabeta(hits, majors, prior_a, prior_b)

print('alpha: {}, beta: {}'.format(a, b))
o = evaluate_odds(2, 12, a, b)
print("  * {}".format(comment_odds(o)))
```

    alpha: 4, beta: 17
    * Probability of h1 with alpha 4 and beta 17: 3.477873773945179e-05
    * Probability of having all the events uniforally distributed with the same alpha/beta: 1.0986756028351329e-05
    * Probability that H1 explains more observed data than H2: 3.165514702402169
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

a, b = evaluate_alphabeta(hits, majors, prior_a, prior_b)
print('alpha: {}, beta: {}'.format(a, b))
k1 = evaluate_odds(2, 12, a, b)
print("  * {}".format(comment_odds(k1)))
```

    alpha: 69, beta: 389
    * Probability of h1 with alpha 69 and beta 389: 3.206705015102759e-85
    * Probability of having all the events uniforally distributed with the same alpha/beta: 6.866633006340007e-90
    * Probability that H1 explains more observed data than H2: 46699.81652058567
      * Overwhelming evidence


Overwhelming evidences? Really? What if we try with other months?


```python
majors = list(df_eq[df_eq['Magnitude'] >= magnitude]['Month'])
hits = [1, 6] # the hypothesis is that earthquakes happens more often in March and September

a, b = evaluate_alphabeta(hits, majors, prior_a, prior_b)
print('alpha: {}, beta: {}'.format(a, b))
k2 = evaluate_odds(2, 12, a, b)
print("  * {}".format(comment_odds(k2)))

```

    alpha: 63, beta: 395
    * Probability of h1 with alpha 63 and beta 395: 5.010476586098064e-81
    * Probability of having all the events uniforally distributed with the same alpha/beta: 1.216465923534471e-83
    * Probability that H1 explains more observed data than H2: 411.88795256508325
      * Overwhelming evidence


Oh oh, January and June also provide overwhelming evidences even better suite the data we have. This tells us that we can't simply blindly say "Yeah, Cosmic Radiations, mainly happening in March and September, causes earthquakes". We need more tests.


```python
print("k1 vs k2 ({}): {}".format(k1/k2, comment_odds(k1/k2)))
print("k2 vs k1 ({}): {}".format(k2/k1, comment_odds(k2/k1)))
```

    k1 vs k2 (113.37990399999995): Strong evidence in favor of H1
    k2 vs k1 (0.008819905157090276): Nothing interesting


Hummm, k1 (Hypothesis 1 with kaggle data) seems far superior over k2. How about the other months? Testing all of them by hand would be quite bothering. 

## Finding the best hypotheses using Monte Carlo Simulations

The cell below is probably way too complicated for what it needs to be, but because of the large quantity of earthquakes in the Kaggle database, running the `evaluate_odds` function on all will case overflows. To go around this, I run 1000 simulations of 400 samples for each pairs of months, then I make an average of the results to evaluate the more likely value. Since there are only 450 earthquakes of 7.2 and higher, it should tend to the average relatively quickly. Still, we can later run this simulation on smaller earthquakes as well.


```python
def monte_carlo_simulation(simulation, sample, lists, prior_a, prior_b):
    outcome = {}
    print('Checking our hypothesis against {} simulations of {} random samples of earthquakes >= {} ({})'.format(simulation, sample, magnitude, len(list(df_eq[df_eq['Magnitude'] >= magnitude]['Month']))))
    print('My priors are that there is a {} chance against {} that there is a relation between earthquakes and months'.format(prior_a, prior_b))
    for x in range(1, 12):
        for y in range(x+1, 13):
            candidates = [x, y]
            xy = ",".join([str(x), str(y)])

            for i in range(simulation):
                earthquakes = np.random.choice(lists, sample)

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

monte_carlo_simulation(simulation, sample, list(df_eq[df_eq['Magnitude'] >= magnitude]['Month']), prior_a, prior_b)
```

    Checking our hypothesis against 1000 simulations of 400 random samples of earthquakes >= 7.2 (447)
    My priors are that there is a 1 chance against 10 that there is a relation between earthquakes and months
    Average of the odds from all the pair of months: 1.2284758425423486e+16: Overwhelming evidence
    5.779371685352091e+17: ['8,11']: Overwhelming evidence
    9.342710423650088e+16: ['7,8']: Overwhelming evidence
    8.16286301178623e+16: ['8,10']: Overwhelming evidence
    1.9752444587255828e+16: ['4,8']: Overwhelming evidence
    1.5072594329784692e+16: ['7,10']: Overwhelming evidence
    1.0061870558221028e+16: ['10,11']: Overwhelming evidence
    6937988395712142.0: ['8,12']: Overwhelming evidence
    2064518783866544.5: ['10,12']: Overwhelming evidence
    929426757591021.6: ['7,11']: Overwhelming evidence
    830093137109461.6: ['5,8']: Overwhelming evidence


Now they are all `Overwhelming evidence`, even the average. This tells us one thing useful: if the sample size is too great, this method of comparing odds degenerates. 

Running the simulation on a smaller sample gives more reasonable results, but it doesn't change the fact that all results are still quite similar, and not singling out one hypothesis.


```python
simulation = 1000
sample = 10

monte_carlo_simulation(simulation, sample, list(df_eq[df_eq['Magnitude'] >= magnitude]['Month']), prior_a, prior_b)
```

    Checking our hypothesis against 1000 simulations of 10 random samples of earthquakes >= 7.2 (447)
    My priors are that there is a 1 chance against 10 that there is a relation between earthquakes and months
    Average of the odds from all the pair of months: 1.8675423313931643: Interesting, but nothing conclusive
    2.6059439277078655: ['8,12']: Interesting, but nothing conclusive
    2.5572177606518767: ['8,11']: Interesting, but nothing conclusive
    2.431827053263679: ['7,8']: Interesting, but nothing conclusive
    2.3995761673128357: ['8,10']: Interesting, but nothing conclusive
    2.391082104372601: ['7,12']: Interesting, but nothing conclusive
    2.339472435488872: ['5,7']: Interesting, but nothing conclusive
    2.33555918007478: ['10,11']: Interesting, but nothing conclusive
    2.3313243407117774: ['7,11']: Interesting, but nothing conclusive
    2.284441807471236: ['3,8']: Interesting, but nothing conclusive
    2.2414130531671916: ['10,12']: Interesting, but nothing conclusive


All the simulation end up with the same evaluation `Interesting, but not conclusive`, including the average. 

From the data available, it doesn't seems like Cosmic Radiations have a significant impact on earthquakes.

## Hypothesis 2: There is a higher chance of earthquakes during typhoons.

The second hypothesis was that Typhoon could cause earthquakes. To test this hypothesis, I used the [Hurricaine dataset from Kaggle](https://www.kaggle.com/noaa/hurricane-database). 

Since there are more earthquakes in the Pasific than the Atlantic ocean, I limited the longitude between Bangkok and the Marshall Islands. This is arbitray, but considering the size of typhoons and their strenght, it could be arguable that their effect could extend on long distances.

First, we need to load the dataset, fix the dates, longitudes and latitudes to match the Earthquake dataset.


```python
df_typhoon = pd.read_csv('typhoon-pacific.csv')

df_typhoon['Month'] = 0
longitude_min = 129 #Bangkook
longitude_max = 163 #Marshall Islands


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
df_typhoon_period = df_typhoon[(df_typhoon['Longitude'] >= 129) & (df_typhoon['Longitude'] <= 163) & (df_typhoon['Date'] >= min(df_eq['Date']))]
df_eq_period = df_eq[(df_eq['Longitude'] >= 129) & (df_eq['Longitude'] <= 163) & (df_eq['Date'] <= max(df_typhoon['Date']))]
```

With the data prepared, I can prepare the values to run the same tests as for the earthquakes and cosmic radiations.

* condition for a: There is an earthquake the same day as a typhoon
* condition for b: There is a typhoon without an earthquake

but before starting, I want to update my prior: I do think that it is unlikely that typhoon causes earthquakes. 1:100 seems like reasonable prior odds


```python
prior_a = 1
prior_b = 100


typhoons = df_typhoon_period['Date'].unique()
earthquakes = df_eq_period[(df_eq_period['Magnitude'] >= magnitude)]['Date'].unique()

e_hits = df_eq_period[(df_eq_period['Date'].isin(list(typhoons))) & (df_eq_period['Magnitude'] >= magnitude)]['Date'].unique()
e_misses = df_eq_period[(~df_eq_period['Date'].isin(list(typhoons))) & (df_eq_period['Magnitude'] >= magnitude)]['Date'].unique()

t_hits = df_typhoon_period[df_typhoon_period['Date'].isin(earthquakes)]['Date'].unique()
t_misses = df_typhoon_period[~df_typhoon_period['Date'].isin(earthquakes)]['Date'].unique()

a, b = evaluate_alphabeta(e_hits, typhoons, prior_a, prior_b)
print("{} earthquakes, {} collisions with typhoons, {} misses".format(len(earthquakes), len(e_hits), len(e_misses)))
print("{} typhoons, {} collisions with earthquakes, {} misses".format(len(typhoons), len(t_hits), len(t_misses)))


print('\nalpha: {}, beta: {}'.format(a, b))
# print('\n{} earthquakes during the period, {} typhoon during the same period, a: {}, b: {}'.format(len(earthquakes), len(typhoons), a, b))


k3 = evaluate_odds(len(e_hits), len(typhoons), a, b)
print("Bayes factor between the hypothesis and randomness: {}: {}".format(k3, comment_odds(k3)))
```

    111 earthquakes, 7 collisions with typhoons, 104 misses
    1335 typhoons, 7 collisions with earthquakes, 1328 misses
    
    alpha: 8, beta: 1428
    * Probability of h1 with alpha 8 and beta 1428: 3.137169035292566e-22
    * Probability of having all the events uniforally distributed with the same alpha/beta: 3.399594319915749e-26
    * Probability that H1 explains more observed data than H2: 9228.06882254796
    Bayes factor between the hypothesis and randomness: 9228.06882254796: Overwhelming evidence


`Overwhelming evidences`? This could also be distorted by the large sample, but opposed to the correlation with the months, there is little space for mistake here. Are we really on something here?

If we do the same calculation, but with typhoon colliding with earthquakes. In both cases, we know that there are 7 collisions, but because there is less strong earthquakes than typhoons, the odds should be more significatif:


```python
a, b = evaluate_alphabeta(t_hits, earthquakes, prior_a, prior_b)

print('alpha: {}, beta: {}'.format(a, b))

k4 = evaluate_odds(len(t_hits), len(earthquakes), a, b)
print("Bayes factor between the hypothesis and randomness: {}: {}".format(k4, comment_odds(k4)))
```

    alpha: 8, beta: 204
    * Probability of h1 with alpha 8 and beta 204: 4.2375421814114814e-16
    * Probability of having all the events uniforally distributed with the same alpha/beta: 6.849143564182111e-18
    * Probability that H1 explains more observed data than H2: 61.869665042092116
    Bayes factor between the hypothesis and randomness: 61.869665042092116: Strong evidence in favor of H1


Well, from the datasets we have, longitude restrictions, and the earthquake size I chose, it seems that there might be `Strong evidences` that typhoons might influence earthquakes.

... I am so surprised. Are we really on to something?
