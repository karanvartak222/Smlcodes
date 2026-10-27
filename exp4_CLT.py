import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data.csv")

# Select Marks column
population = data["Marks"]

# Different sample sizes
sample_sizes = [5, 10, 30, 50]

# Number of samples
num_samples = 1000

plt.figure(figsize=(12, 8))

for i, n in enumerate(sample_sizes):
    sample_means = []

    for j in range(num_samples):
        sample = population.sample(
            n=n,
            replace=True
        )
        sample_means.append(sample.mean())

    plt.subplot(2, 2, i + 1)
    plt.hist(
        sample_means,
        bins=30,
        edgecolor="black"
    )

    plt.title(f"Sample Size = {n}")
    plt.xlabel("Sample Mean")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()



Marks
42
55
68
35
72
49
61
58
45
66
52
39
75
57
63
48
54
69
41
60
51
64
47
70
56
44
62
53
67
38
73
50
59
46
65
43
71
55
60
52
40
68
57
49
63
54
74
45
61
58
