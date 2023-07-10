import numpy as np

def count_occurrences(array, row, col, condition):
    count_above = 0
    count_below = 0
    count_left = 0
    count_right = 0

    # Count occurrences above
    for r in range(row-1, -1, -1):
        if array[r][col] == 1:
            break
        if array[r][col] in condition:
            count_above += 1

    # Count occurrences below
    for r in range(row+1, array.shape[0]):
        if array[r][col] == 1:
            break
        if array[r][col] in condition:
            count_below += 1

    # Count occurrences to the left
    for c in range(col-1, -1, -1):
        if array[row][c] == 1:
            break
        if array[row][c] in condition:
            count_left += 1

    # Count occurrences to the right
    for c in range(col+1, array.shape[1]):
        if array[row][c] == 1:
            break
        if array[row][c] in condition:
            count_right += 1

    return count_above, count_below, count_left, count_right
# Create a sample 2D NumPy array
array = np.array([[1, 2, 3, -1],
                  [4, 0, -1, 4],
                  [-1, 8, 9, 10]])

# Count occurrences above, below, left, and right of cell (1, 1) that are greater than 4
count_above, count_below, count_left, count_right = count_occurrences(array, 1, 1, [4])
print("Count above:", count_above)
print("Count below:", count_below)
print("Count left:", count_left)
print("Count right:", count_right)