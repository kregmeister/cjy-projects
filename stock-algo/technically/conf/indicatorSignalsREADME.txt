General rules:
- A feature cannot be a key, only keywords (handled explicitly by conditional logic) can be keys
- A keyword cannot be a value, only a key
- The indicator key serves as an organizational marker; it is not used in python logic file
- When length or direction is specified, it must be located directly after the clause in which it references
- When a signal has multiple parameters, and (&) is implied. For or (|), consecutive conditions in a multi condition must end with digits
- IMPORTANT: If two conditions may be mutually inclusive, put the 'rarer' (hypothetically stronger) one first.
- If a condition type has multiple instances, the keys can be unique:
  "divergence": {
      "coFeature": {
          "close": "75th"
      },
      "direction": 1,
      "currentTrend": 1
  }
  "divergenceLong": {
      "coFeature": {
          "close": "75th"
      },
      "length": 5
  }
- To simulate an or (|) clause, set the suffix of second and beyond clause to a SINGLE digit:
"crossover": {
    "above": 0,
    "currentTrend": 1
},
"crossover2": {
    "below": 0,
    "currentTrend": 1
}
- Do not use single digit suffixes if you don't want them to be or 


These are the rules for specifying technical indicator (feature) signal circumstances/parameters:
- threshold: Options (above or below: integer or percentile)
- crossover: Options (above or below: coFeature or integer) 
  --- where coFeature is the column to compare
- divergence: Options (coFeature or coFeatureTrend: minMaxDiff (percentile)) 
  --- where minMaxDiff is scaled abs(feature - coFeature)
  --- where coFeatureTrend is scaled abs(featureTrend - coFeatureTrend)
- convergence: Options (coFeature or coFeatureTrend: minMaxDiff (percentile))
- direction (exclusive to divergence): if 1, then feature diverging above cofeature. Else, vice-versa.
- multi: Options (Can contain any of the above)

ALL options can have:
- circumstance: Options(currentTrend: 1,0,-1, length: int)
  --- where length is how many periods in a row the condition has been met