#BCI
import numpy as np
import matplotlib.pyplot as plt

# Reproducible results
np.random.seed(42)

# Observed sample data
data = np.array([5, 6, 7, 7, 8, 9, 10, 6, 7, 8])

# Original sample mean
original_mean = np.mean(data)

# Number of bootstrap resamples
B = 10000

# Generate bootstrap means
bootstrap_means = []

for _ in range(B):
    bootstrap_sample = np.random.choice(
        data,
        size=len(data),
        replace=True
    )
    
    bootstrap_means.append(
        np.mean(bootstrap_sample)
    )

bootstrap_means = np.array(bootstrap_means)

# 95% Bootstrap Confidence Interval
lower = np.percentile(bootstrap_means, 2.5)
upper = np.percentile(bootstrap_means, 97.5)

# Display results
print("Original sample mean:", original_mean)

print("95% Bootstrap Confidence Interval:")
print("Lower limit:", lower)
print("Upper limit:", upper)

# Plot bootstrap distribution
plt.hist(bootstrap_means, bins=30)

plt.axvline(
    original_mean,
    linestyle="--",
    label="Original Mean"
)

plt.axvline(
    lower,
    linestyle="--",
    label="Lower 2.5%"
)

plt.axvline(
    upper,
    linestyle="--",
    label="Upper 97.5%"
)

plt.xlabel("Bootstrap Mean")
plt.ylabel("Frequency")
plt.title("Bootstrap Distribution of the Mean")

plt.legend()
plt.show()


#PT
import numpy as np
import matplotlib.pyplot as plt

# Reproducible results
np.random.seed(42)

# Two independent groups
group_A = np.array([60, 62, 65, 63, 61])
group_B = np.array([70, 72, 68, 71, 69])

# Observed difference in means
observed_difference = np.mean(group_A) - np.mean(group_B)

# Combine observations
combined = np.concatenate([group_A, group_B])

# Number of permutations
B = 10000

permutation_differences = []

n_A = len(group_A)

# Generate permutations
for _ in range(B):
    shuffled = np.random.permutation(combined)

    shuffled_A = shuffled[:n_A]
    shuffled_B = shuffled[n_A:]

    difference = np.mean(shuffled_A) - np.mean(shuffled_B)

    permutation_differences.append(difference)

permutation_differences = np.array(permutation_differences)

# Two-sided permutation p-value
p_value = np.mean(
    np.abs(permutation_differences) >= abs(observed_difference)
)

# Display results
print("Observed difference in means:", observed_difference)
print("Permutation p-value:", p_value)

# Decision
alpha = 0.05

if p_value < alpha:
    print("Decision: Reject H0")
    print("Conclusion: There is significant evidence of a difference between the group means.")
else:
    print("Decision: Do not reject H0")
    print("Conclusion: There is not enough evidence of a difference between the group means.")

# Plot null distribution
plt.hist(permutation_differences, bins=30)

plt.axvline(
    observed_difference,
    linestyle="--",
    label="Observed Difference"
)

plt.xlabel("Difference in Means")
plt.ylabel("Frequency")
plt.title("Permutation Null Distribution")

plt.legend()
plt.show()
